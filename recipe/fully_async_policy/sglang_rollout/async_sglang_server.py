# Copyright 2023-2024 SGLang Team
# Copyright 2025 Infrawaves Ltd. and/or its affiliates
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
import logging
import asyncio
from typing import Any, Optional, Sequence
import ray
import torch
import sglang.srt.entrypoints.engine
import copy
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig, RewardModelConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangHttpServerBase,
    SGLangReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

@ray.remote(num_cpus=1)
class SGLangHttpServerForPartial(SGLangHttpServerBase):
    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, nnodes, cuda_visible_devices)

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}  # request_id -> Event
        self.req_output: dict[str, Optional[dict[str, Any]]] = {}

    async def _generate_step(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = True

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )

        output = await self.tokenizer_manager.generate_request(request, None).__anext__()

        self.req_output[request_id] = output
        assert self.req_output[request_id] is not None

    async def generate_for_partial(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    )-> tuple[list[Any], list[Any], bool] | tuple[Sequence[int], list[float], Any]:
        async with self.lock:
            if self.paused:
                return [], [], True  # indicate cancelled
            self.req_output[request_id] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            sampling_params.pop('logprobs', None)
            # Convert prompt_ids from list[int] to GPU tensor
            # prompt_ids_tensor = torch.tensor(prompt_ids, device="cuda").unsqueeze(0)
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data)
            )

        done, pend = await asyncio.wait([generation_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            await task

        for task in pend:
            task.cancel()

        async with self.lock:
            output = self.req_output[request_id]
            if output is None:
                return [], [], True  # indicate cancelled
            
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                return [], [], True  # indicate cancelled

            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )

            is_cancel = generation_handle not in done
            self.cancel_event.pop(request_id, None)
            self.req_output.pop(request_id, None)
        return  list(token_ids), list(log_probs), is_cancel

    async def cancel(self):
        # async with self.lock:
        #    self.paused = True
        #    for request_id in self.cancel_event:
        #        self.cancel_event[request_id].set()
        async with self.lock:
            self.tokenizer_manager.abort_request(abort_all=True)

    async def resume(self):
        async with self.lock:
            self.paused = False

    async def reset_prefix_cache(self):
        async with self.lock:
            self.tokenizer_manager.abort_request(abort_all=True)

class FullyAsyncSGLangReplica(SGLangReplica):
    def __init__(
        self, 
        replica_rank: int, 
        config: RolloutConfig | RewardModelConfig, 
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node)
        self.server_class = SGLangHttpServerForPartial
        
    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])

    async def reset_prefix_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.reset_prefix_cache.remote() for server in self.servers])
