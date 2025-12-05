# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import time
from pprint import pformat
from typing import Coroutine, Any, Tuple, List, Dict, Optional
import copy

import ray
from ray import ObjectRef

from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    merge_rollout_sample,
    prepare_single_generation_data,
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger

from uuid import uuid4
from recipe.fully_async_policy.order_set import OrderedSet

import aio.Scheduler.config as scheduler_config
import aiohttp

@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.global_steps = 0
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.strong_paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Initialize async locks directly
        self.lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)

        # temp rollout samples
        self.temp_rollout_samples: Dict[str, List] = {}
        self.temp_rollout_samples_lock = asyncio.Lock()
        self.temp_rollout_staleness_samples: int = 0

        self.using_sample_lock = asyncio.Lock()
        self.using_rollout_sample = None
        self.using_sample_index = 0 

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples / len(self.async_rollout_manager.server_handles))
            self.max_concurrent_requests = self.max_concurrent_samples * self.config.actor_rollout_ref.rollout.n
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
                f"max_concurrent_requests: {self.max_concurrent_requests}"
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                sum(len(tasks) for tasks in self.active_tasks)
                + sum(len(tasks) for tasks in self.interaction_tasks)
                + self.result_queue.qsize() * self.config.actor_rollout_ref.rollout.n
                + sum(queue.qsize() for queue in self.cancel_queue)
                + (await self.message_queue_client.get_queue_size()) * self.config.actor_rollout_ref.rollout.n
                + self.temp_rollout_staleness_samples
                + sum(queue.qsize() for queue in self.before_interaction_queue)
                + sum(queue.qsize() for queue in self._potential_after_interaction_queue)
                + sum(queue.qsize() for queue in self._official_after_interaction_queue)
            )
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            val_metrics = None
            if (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0  # don't test here in the initial parameter sync
            ) or (validate and self.val_reward_fn is not None):
                with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

            self.version_start_time = time.time()

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

        self.active_tasks = [set() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.interaction_tasks = [set() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.result_queue = asyncio.Queue()
        self.cancel_queue = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.priority_cancel_queue = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.is_priority_cancel_set = [set() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.is_priority_cancel_set_lock = [asyncio.Lock() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.before_interaction_queue = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]
        self._potential_after_interaction_queue = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]
        self._official_after_interaction_queue = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]

        self.worker_lock = [asyncio.Lock() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.condition = [asyncio.Condition(self.worker_lock[i]) for i in range(len(self.async_rollout_manager.server_handles))]

        # Used for recording request sequence
        self.is_official_candidate_sample = [{} for _ in range(len(self.async_rollout_manager.server_handles))]
        self.official_candidate_sample_cnt = [0 for _ in range(len(self.async_rollout_manager.server_handles))]
        self.potential_candidate_samples = [asyncio.Queue() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.is_generation_order_set = [OrderedSet() for _ in range(len(self.async_rollout_manager.server_handles))]
        self.is_generation_order_set_lock = [asyncio.Lock() for _ in range(len(self.async_rollout_manager.server_handles))]

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(
                batch_dict, self.global_steps, self.config.actor_rollout_ref.rollout.n
            )

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                rollout_status={},
                request_id=""
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _processor_worker(self, server_index: int):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                try:
                    await self._pause_worker_v3(server_index)
                except Exception as e:
                    raise RuntimeError(f"Exception occurred while pausing worker: {e}")

                async with self.worker_lock[server_index]:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition[server_index].wait()
                continue

            simple_from_cancel_queue = False
            simple_from_official_after_interaction_queue = False
            simple_from_potential_after_interaction_queue = False
            simple_from_priority_cancel_queue = False
            if not self._official_after_interaction_queue[server_index].empty():
                rollout_sample, request_index = await self._official_after_interaction_queue[server_index].get()
                simple_from_official_after_interaction_queue = True
            elif not self.cancel_queue[server_index].empty():
                simple_from_cancel_queue = True
                rollout_sample, request_index = await self.cancel_queue[server_index].get()
            elif not self._potential_after_interaction_queue[server_index].empty():
                simple_from_potential_after_interaction_queue = True
            elif not self.priority_cancel_queue[server_index].empty():
                simple_from_priority_cancel_queue = True

            # Check whether the number of concurrent tasks exceeds the limit
            if simple_from_official_after_interaction_queue or simple_from_potential_after_interaction_queue or simple_from_cancel_queue or simple_from_priority_cancel_queue:
                if len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                    # Check the request is official candidate sample or potential one
                    if (simple_from_official_after_interaction_queue or simple_from_cancel_queue) and self.is_official_candidate_sample[server_index].get(rollout_sample.request_id, False):
                        # If it's an official candidate sample, it can abort an potential one
                        async with self.is_generation_order_set_lock[server_index]:
                            abort_request_id = self.is_generation_order_set[server_index].find_first_potential_candidate_sample(self.is_official_candidate_sample[server_index])
                        if abort_request_id is not None:
                            async with self.is_generation_order_set_lock[server_index]:
                                self.is_generation_order_set[server_index].discard(abort_request_id)
                            async def _wait_ref(ref):
                                return await ref
                            abort_handle = asyncio.create_task(_wait_ref(
                                self.async_rollout_manager.server_handles[server_index].cancel_request.remote(abort_request_id)
                            ))
                            async with self.is_priority_cancel_set_lock[server_index]:
                                self.is_priority_cancel_set[server_index].add(abort_request_id)
                        else:
                            while len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                                async with self.worker_lock[server_index]:
                                    if self.active_tasks[server_index]:
                                        done_tasks, self.active_tasks[server_index] = await asyncio.wait(
                                            self.active_tasks[server_index], return_when=asyncio.FIRST_COMPLETED
                                        )
                                    for task in done_tasks:
                                        await task
                    else:
                        while len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                            async with self.worker_lock[server_index]:
                                if self.active_tasks[server_index]:
                                    done_tasks, self.active_tasks[server_index] = await asyncio.wait(
                                        self.active_tasks[server_index], return_when=asyncio.FIRST_COMPLETED
                                    )
                                for task in done_tasks:
                                    await task

                            if not simple_from_official_after_interaction_queue and not simple_from_cancel_queue: 
                                if not self._official_after_interaction_queue[server_index].empty():
                                    rollout_sample, request_index = await self._official_after_interaction_queue[server_index].get()
                                    simple_from_official_after_interaction_queue = True
                                    simple_from_potential_after_interaction_queue = False
                                    simple_from_priority_cancel_queue = False
                                    simple_from_cancel_queue = False
                                    break
                                elif not self.cancel_queue[server_index].empty():
                                    simple_from_cancel_queue = True
                                    rollout_sample, request_index = await self.cancel_queue[server_index].get()
                                    simple_from_official_after_interaction_queue = False
                                    simple_from_potential_after_interaction_queue = False
                                    simple_from_priority_cancel_queue = False
                                    break
            else:
                while len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                    async with self.worker_lock[server_index]:
                        if self.active_tasks[server_index]:
                            done_tasks, self.active_tasks[server_index] = await asyncio.wait(
                                self.active_tasks[server_index], return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                    # double check if there are interaction_tasks can be used
                    if not self._official_after_interaction_queue[server_index].empty():
                        rollout_sample, request_index = await self._official_after_interaction_queue[server_index].get()
                        simple_from_official_after_interaction_queue = True
                        break
                    elif not self.cancel_queue[server_index].empty():
                        simple_from_cancel_queue = True
                        rollout_sample, request_index = await self.cancel_queue[server_index].get()
                        break
                    elif not self._potential_after_interaction_queue[server_index].empty():
                        simple_from_potential_after_interaction_queue = True
                        break
                    elif not self.priority_cancel_queue[server_index].empty():
                        simple_from_priority_cancel_queue = True
                        break

            if simple_from_potential_after_interaction_queue:
                rollout_sample, request_index = await self._potential_after_interaction_queue[server_index].get()
            elif simple_from_priority_cancel_queue:
                rollout_sample, request_index = await self.priority_cancel_queue[server_index].get()

            if not simple_from_cancel_queue and not simple_from_priority_cancel_queue and not simple_from_official_after_interaction_queue and not simple_from_potential_after_interaction_queue:
                async with self.using_sample_lock:
                    request_index = self.using_sample_index
                    if self.using_sample_index == 0:
                        rollout_sample = await self.pending_queue.get()
                        self.using_rollout_sample = copy.deepcopy(rollout_sample)
                        self.using_sample_index += 1
                    elif self.using_sample_index == self.config.actor_rollout_ref.rollout.n - 1:
                        self.using_sample_index = 0
                        rollout_sample = copy.deepcopy(self.using_rollout_sample)
                    else:
                        rollout_sample = copy.deepcopy(self.using_rollout_sample)
                        self.using_sample_index += 1
                    self.staleness_samples += 1
                    rollout_sample.request_id = str(uuid4())
                    # Check if this task is official candidate sample or potential one
                    if self.official_candidate_sample_cnt[server_index] < self.max_concurrent_requests:
                        self.is_official_candidate_sample[server_index][rollout_sample.request_id] = True
                        async with self.is_generation_order_set_lock[server_index]:
                            self.official_candidate_sample_cnt[server_index] += 1
                    else:
                        self.is_official_candidate_sample[server_index][rollout_sample.request_id] = False
                        await self.potential_candidate_samples[server_index].put(rollout_sample.request_id)

            if not simple_from_cancel_queue and rollout_sample == "DONE":
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                await self._pause_worker_v3(server_index)
                break

            # Submit single sample processing
            if self.paused:
                try:
                    await self._pause_worker_v3(server_index)
                except Exception as e:
                    raise RuntimeError(f"Exception occurred while pausing worker: {e}")

                async with self.worker_lock[server_index]:
                    while self.paused:
                        await self.condition[server_index].wait()

            async with self.worker_lock[server_index]:  
                task = asyncio.create_task(
                    self._process_single_request_streaming(rollout_sample, request_index, server_index),
                    name=f"{rollout_sample.sample_id}:{request_index}",
                )
                setattr(task, "_kind", "active_task")
                self.active_tasks[server_index].add(task)
            async with self.is_generation_order_set_lock[server_index]:
                self.is_generation_order_set[server_index].add(rollout_sample.request_id, not self.is_official_candidate_sample[server_index][rollout_sample.request_id])

            if simple_from_cancel_queue:
                self.cancel_queue[server_index].task_done()
            elif simple_from_official_after_interaction_queue:
                self._official_after_interaction_queue[server_index].task_done()
            elif simple_from_potential_after_interaction_queue:
                self._potential_after_interaction_queue[server_index].task_done()
            elif simple_from_priority_cancel_queue:
                self.priority_cancel_queue[server_index].task_done()
            else:
                if request_index == 0:
                    self.pending_queue.task_done()

    async def _processor_interaction_worker(self, server_index: int):
        """
        Interaction worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            # Get the rollout sample and request index from the queue
            # No need to wait for task to complete
            rollout_sample, request_index = await self.before_interaction_queue[server_index].get()
            async with self.worker_lock[server_index]:
                task = asyncio.create_task(self._process_single_request_streaming(rollout_sample, request_index, server_index))
                setattr(task, "_kind", "interaction_task")
                self.interaction_tasks[server_index].add(task)
            self.before_interaction_queue[server_index].task_done()

    async def _process_single_request_streaming(self, rollout_sample: RolloutSample, request_index: int, server_index: int):
        """Process a single request streamingly"""
        # Calling asynchronous generation methods
        partial_output = rollout_sample.agent_loop_output_list[request_index]
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        agent_loop_output = await self.async_rollout_manager.generate_single_request_async(
            rollout_sample.full_batch, request_index, server_index, rollout_sample.request_id, partial_output
        )

        async with self.temp_rollout_samples_lock:
            if rollout_sample.sample_id not in self.temp_rollout_samples:
                self.temp_rollout_samples[rollout_sample.sample_id] = [rollout_sample, 0]
            self.temp_rollout_samples[rollout_sample.sample_id][0].agent_loop_output_list[request_index] = agent_loop_output

        is_cancel = agent_loop_output.is_cancel
        is_processing_tools = agent_loop_output.is_processing_tools
        is_interaction = agent_loop_output.is_interaction
        is_after_interacting = agent_loop_output.is_after_interacting

        # Discard request from is_generation_order_set
        async with self.is_generation_order_set_lock[server_index]:
            if not is_after_interacting:
                self.is_generation_order_set[server_index].discard(rollout_sample.request_id)

        if is_cancel:
            # Put in the cancel queue and wait for the generation to resume
            rollout_sample.agent_loop_output_list[request_index] = agent_loop_output
            if self.is_priority_cancel_set[server_index].__contains__(rollout_sample.request_id):
                async with self.is_priority_cancel_set_lock[server_index]:
                    self.is_priority_cancel_set[server_index].remove(rollout_sample.request_id)
                # Put into priority cancel queue
                await self.priority_cancel_queue[server_index].put([rollout_sample, request_index])
            else:
                await self.cancel_queue[server_index].put([rollout_sample, request_index])
        elif is_processing_tools or is_interaction:
            # Remove when not successfully be aborted
            if self.is_priority_cancel_set[server_index].__contains__(rollout_sample.request_id):
                async with self.is_priority_cancel_set_lock[server_index]:
                    self.is_priority_cancel_set[server_index].remove(rollout_sample.request_id)
            rollout_sample.agent_loop_output_list[request_index] = agent_loop_output
            await self.before_interaction_queue[server_index].put([rollout_sample, request_index])
        elif is_after_interacting:
            # Put into after_interaction_queue
            rollout_sample.agent_loop_output_list[request_index] = agent_loop_output
            if self.is_official_candidate_sample[server_index].get(rollout_sample.request_id, False):
                await self._official_after_interaction_queue[server_index].put([rollout_sample, request_index])
            else:
                await self._potential_after_interaction_queue[server_index].put([rollout_sample, request_index])
        else:
            async with self.temp_rollout_samples_lock:
                self.temp_rollout_samples[rollout_sample.sample_id][1] += 1
                self.temp_rollout_staleness_samples += 1

            # Remove when not successfully be aborted
            if self.is_priority_cancel_set[server_index].__contains__(rollout_sample.request_id):
                async with self.is_priority_cancel_set_lock[server_index]:
                    self.is_priority_cancel_set[server_index].remove(rollout_sample.request_id)

            # Change is_official_candidate_sample
            async with self.is_generation_order_set_lock[server_index]:
                if self.is_official_candidate_sample[server_index].get(rollout_sample.request_id, False):
                    is_fill_new_official = False
                    while not self.potential_candidate_samples[server_index].empty():
                        next_request_id = await self.potential_candidate_samples[server_index].get()
                        if next_request_id in self.is_official_candidate_sample[server_index]:
                            self.is_official_candidate_sample[server_index][next_request_id] = True
                            is_fill_new_official = True
                            self.potential_candidate_samples[server_index].task_done()
                            break
                        self.potential_candidate_samples[server_index].task_done()
                    if not is_fill_new_official:
                        self.official_candidate_sample_cnt[server_index] -= 1
                else:
                    self.official_candidate_sample_cnt[server_index] -= 1
                del self.is_official_candidate_sample[server_index][rollout_sample.request_id]

            if self.temp_rollout_samples[rollout_sample.sample_id][1] == self.config.actor_rollout_ref.rollout.n:
                # put into the result_queue
                rollout_sample = self.temp_rollout_samples[rollout_sample.sample_id][0]
                rollout_sample.param_version = self.current_param_version
                rollout_sample.rollout_status = await self.get_statistics()
                await self.result_queue.put(rollout_sample)
                async with self.temp_rollout_samples_lock:
                    del self.temp_rollout_samples[rollout_sample.sample_id]
                    self.temp_rollout_staleness_samples -= self.config.actor_rollout_ref.rollout.n

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        agent_loop_output_list = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        rollout_sample.agent_loop_output_list = agent_loop_output_list

        is_cancel = False
        for agent_loop in agent_loop_output_list:
            if not is_cancel and agent_loop.is_cancel:
                is_cancel = True

        if is_cancel:
            # Put in the cancel queue and wait for the generation to resume
            await self.cancel_queue.put(rollout_sample)
        else:
            # put into the result_queue
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            await self.result_queue.put(rollout_sample)

        self.processed_sample_count += 1

    async def _pause_worker_v3(self, server_index: Optional[int] = None):
        while self.active_tasks[server_index] or self.interaction_tasks[server_index]:
            if self.strong_paused:
                break
            simple_from_official_after_interaction_queue = False
            simple_from_potential_after_interaction_queue = False
            simple_from_priority_cancel_queue = False
            if not self._official_after_interaction_queue[server_index].empty():
                rollout_sample, request_index = await self._official_after_interaction_queue[server_index].get()
                simple_from_official_after_interaction_queue = True
            elif not self._potential_after_interaction_queue[server_index].empty():
                simple_from_potential_after_interaction_queue = True
            elif not self.priority_cancel_queue[server_index].empty():
                simple_from_priority_cancel_queue = True
            if not simple_from_official_after_interaction_queue and not simple_from_potential_after_interaction_queue and not simple_from_priority_cancel_queue:
                async with self.worker_lock[server_index]:
                    all_tasks = self.active_tasks[server_index] | self.interaction_tasks[server_index]
                    if not all_tasks:
                        break
                    done, pending = await asyncio.wait(
                        all_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            raise RuntimeError(f"Exception in task {task.get_name()}: {e}")
                        finally:
                            kind = getattr(task, "_kind", None)
                            if kind == "active_task":
                                self.active_tasks[server_index].discard(task)
                            elif kind == "interaction_task":
                                self.interaction_tasks[server_index].discard(task)
                continue
            if len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                # Check the request is official candidate sample or potential one
                if simple_from_official_after_interaction_queue and self.is_official_candidate_sample[server_index].get(rollout_sample.request_id, False):
                    # If it's an official candidate sample, it can abort an potential one
                    async with self.is_generation_order_set_lock[server_index]:
                        abort_request_id = self.is_generation_order_set[server_index].find_first_potential_candidate_sample(self.is_official_candidate_sample[server_index])
                    if abort_request_id is not None:
                        async with self.is_generation_order_set_lock[server_index]:
                            self.is_generation_order_set[server_index].discard(abort_request_id)
                        async def _wait_ref(ref):
                            return await ref
                        abort_handle = asyncio.create_task(_wait_ref(
                            self.async_rollout_manager.server_handles[server_index].cancel_request.remote(abort_request_id)
                        ))
                        async with self.is_priority_cancel_set_lock[server_index]:
                            self.is_priority_cancel_set[server_index].add(abort_request_id)
                    else:
                        while len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                            async with self.worker_lock[server_index]:
                                if self.active_tasks[server_index]:
                                    done_tasks, self.active_tasks[server_index] = await asyncio.wait(
                                        self.active_tasks[server_index], return_when=asyncio.FIRST_COMPLETED
                                    )
                                for task in done_tasks:
                                    await task
                else:
                    while len(self.active_tasks[server_index]) >= self.max_concurrent_requests:
                        async with self.worker_lock[server_index]:
                            if self.active_tasks[server_index]:
                                done_tasks, self.active_tasks[server_index] = await asyncio.wait(
                                    self.active_tasks[server_index], return_when=asyncio.FIRST_COMPLETED
                                )
                            for task in done_tasks:
                                await task

                        if not simple_from_official_after_interaction_queue and not self._official_after_interaction_queue[server_index].empty():
                            rollout_sample, request_index = await self._official_after_interaction_queue[server_index].get()
                            simple_from_official_after_interaction_queue = True
                            simple_from_potential_after_interaction_queue = False
                            simple_from_priority_cancel_queue = False
                            break

            if simple_from_potential_after_interaction_queue:
                rollout_sample, request_index = await self._potential_after_interaction_queue[server_index].get()
            elif simple_from_priority_cancel_queue:
                rollout_sample, request_index = await self.priority_cancel_queue[server_index].get()

            # Submit single request processing
            async with self.worker_lock[server_index]:  
                task = asyncio.create_task(
                    self._process_single_request_streaming(rollout_sample, request_index, server_index),
                    name=f"{rollout_sample.sample_id}:{request_index}",
                )
                setattr(task, "_kind", "active_task")
                self.active_tasks[server_index].add(task)
            async with self.is_generation_order_set_lock[server_index]:
                self.is_generation_order_set[server_index].add(rollout_sample.request_id, not self.is_official_candidate_sample[server_index][rollout_sample.request_id])

            if simple_from_official_after_interaction_queue:
                self._official_after_interaction_queue[server_index].task_done()
            elif simple_from_potential_after_interaction_queue:
                self._potential_after_interaction_queue[server_index].task_done()
            elif simple_from_priority_cancel_queue:
                self.priority_cancel_queue[server_index].task_done()

    async def _pause_worker(self, server_index: Optional[int] = None):
        """
        Pause worker coroutine, only put tasks from interaction_tasks to active_tasks continuously
        """
        while self.active_tasks[server_index] or self.interaction_tasks[server_index]:
            async with self.worker_lock[server_index]:
                all_tasks = self.active_tasks[server_index] | self.interaction_tasks[server_index]
                if not all_tasks:
                    break
                done, pending = await asyncio.wait(
                    all_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        raise RuntimeError(f"Exception in task {task.get_name()}: {e}")
                    finally:
                        kind = getattr(task, "_kind", None)
                        if kind == "active_task":
                            self.active_tasks[server_index].discard(task)
                        elif kind == "interaction_task":
                            self.interaction_tasks[server_index].discard(task)

    async def _pause_worker_v2(self, server_index: Optional[int] = None):
        """
        Pause worker coroutine, only put tasks from interaction_tasks to active_tasks continuously
        """
        while self.active_tasks[server_index] or self.interaction_tasks[server_index]:
            async with self.worker_lock[server_index]:
                all_tasks = self.active_tasks[server_index] | self.interaction_tasks[server_index]
                if not all_tasks:
                    break
                done, pending = await asyncio.wait(
                    all_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await task
                    if getattr(task, "_kind", None) == "active_task":
                        self.active_tasks[server_index].remove(task)
                    elif getattr(task, "_kind", None) == "interaction_task":
                        self.interaction_tasks[server_index].remove(task)

    async def _consumer_worker(self):
        """
        The consumer coroutine is responsible for obtaining the processing results
        from the result queue and putting them into the message queue
        """
        while True:
            rollout_sample = await self.result_queue.get()
            rollout_sample = merge_rollout_sample(self.config, self.tokenizer, rollout_sample)

            # Put RolloutSample into the message queue、
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1

            self.result_queue.task_done()

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        # we start from step 1
        self.global_steps += 1

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine and consumer coroutine
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = [asyncio.create_task(self._processor_worker(server_index)) for server_index in range(len(self.async_rollout_manager.server_handles))]
        self.consumer_task = asyncio.create_task(self._consumer_worker())
        self.interaction_task = [asyncio.create_task(self._processor_interaction_worker(server_index)) for server_index in range(len(self.async_rollout_manager.server_handles))]

        tasks = [self.feed_task, self.consumer_task] + self.processor_task + self.interaction_task

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task.exception():
                    raise task.exception()
            # Wait for sample feed to complete
            await self.feed_task
            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await asyncio.gather(*self.processor_task)
            print("[FullyAsyncRollouter] Streaming process completed")

            # Waiting for the result queue to clear
            await self.result_queue.join()
            print("[FullyAsyncRollouter] Result queue cleared")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception:{e}")

        finally:
            if self.processor_task:
                self.processor_task.cancel()
            if self.consumer_task:
                self.consumer_task.cancel()

            await asyncio.gather(*self.processor_task, self.consumer_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.strong_paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.strong_paused = False
                    for server_index in range(len(self.async_rollout_manager.server_handles)):
                        async with self.worker_lock[server_index]:
                            self.condition[server_index].notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples * self.config.actor_rollout_ref.rollout.n:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples * self.config.actor_rollout_ref.rollout.n} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause]")
        async with self.lock:
            self.paused = True
            self.strong_paused = True
            # Cancel all rollout tasks
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel()
            pause_tasks = []
            for server_index in range(len(self.async_rollout_manager.server_handles)):
                if self.active_tasks[server_index] or self.interaction_tasks[server_index]:
                    pause_tasks.append(asyncio.create_task(self._pause_worker_v2(server_index)))
            await asyncio.gather(*pause_tasks, return_exceptions=True)
            for server_index in range(len(self.async_rollout_manager.server_handles)):
                self.active_tasks[server_index].clear()
                self.interaction_tasks[server_index].clear()
            print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")
            await self.async_rollout_manager.reset_prefix_cache()
            self.monitor_loop_trigger = False

    async def start_concurrency_monitor(self):
        """Start concurrency monitor"""
        print("[FullyAsyncRollouter][Public][Start Concurrency Monitor]")
        proxy_url = f"http://{scheduler_config.PROXY_HOST_IP}:{scheduler_config.PROXY_HOST_PORT}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(proxy_url + "/start_monitor_concurrency", timeout=30) as resp:
                    if resp.status != 200:
                        error_msg = f"Error: Failed to start concurrency monitor. Status code: {resp.status}"
                        print(f"[FullyAsyncRollouter] {error_msg}")
            except Exception as e:
                error_msg = f"Error: Exception occurred while starting concurrency monitor: {e}"
                print(f"[FullyAsyncRollouter] {error_msg}")
                raise e

    async def adjust_environment(self):
        """Adjust environment before resuming rollout"""
        print("[FullyAsyncRollouter][Public][Adjust Environment]")
        # First Calculate max new staleness samples allowed
        reset_staleness_samples = (
            sum(len(tasks) for tasks in self.active_tasks)
            + sum(len(tasks) for tasks in self.interaction_tasks)
            + self.result_queue.qsize() * self.config.actor_rollout_ref.rollout.n
            + sum(queue.qsize() for queue in self.cancel_queue)
            + (await self.message_queue_client.get_queue_size())
            * self.config.actor_rollout_ref.rollout.n
            + self.temp_rollout_staleness_samples
            + sum(queue.qsize() for queue in self.before_interaction_queue)
            + sum(queue.qsize() for queue in self._potential_after_interaction_queue)
            + sum(queue.qsize() for queue in self._official_after_interaction_queue)
        )

        # Get proxy url
        # First stop concurrency_monitor then auto-scaling
        proxy_url = f"http://{scheduler_config.PROXY_HOST_IP}:{scheduler_config.PROXY_HOST_PORT}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(proxy_url + "/stop_monitor_concurrency", timeout=30) as resp:
                    if resp.status != 200:
                        error_msg = f"Error: Failed to stop concurrency monitor. Status code: {resp.status}"
                        print(f"[FullyAsyncRollouter] {error_msg}")

                # Now adjust auto-scaling
                async with session.post(
                    proxy_url + "/auto_scaling",
                    params={
                        "max_concurrent_requests": str(
                            self.max_required_samples * self.config.actor_rollout_ref.rollout.n - reset_staleness_samples
                        )
                    },
                    timeout=300,
                ) as adjust_resp:
                    if adjust_resp.status != 200:
                        error_msg = f"Error: Failed to adjust auto-scaling. Status code: {adjust_resp.status}"
                        print(f"[FullyAsyncRollouter] {error_msg}")

                async with session.post(proxy_url + "/start_monitor_concurrency", timeout=30) as start_resp:
                    if start_resp.status != 200:
                        error_msg = f"Error: Failed to start concurrency monitor. Status code: {start_resp.status}"
                        print(f"[FullyAsyncRollouter] {error_msg}")            

            except Exception as e:
                error_msg = f"Error: Exception occurred while adjusting environment: {e}"
                print(f"[FullyAsyncRollouter] {error_msg}")
                raise e

    async def resume(self, dependency_ref: ObjectRef = None):
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            self.paused = False
            self.strong_paused = False
            self.monitor_loop_trigger = True
        for server_index in range(len(self.async_rollout_manager.server_handles)):
            async with self.worker_lock[server_index]:
                self.condition[server_index].notify_all()

            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": sum(len(tasks) for tasks in self.active_tasks),
            "monitor/interaction_tasks_size": sum(len(tasks) for tasks in self.interaction_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/cancel_queue_size": sum(queue.qsize() for queue in self.cancel_queue),
            "monitor/queue/priority_cancel_queue_size": sum(queue.qsize() for queue in self.priority_cancel_queue),
            "monitor/queue/result_queue_size": self.result_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/current_param_version": self.current_param_version,
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats
