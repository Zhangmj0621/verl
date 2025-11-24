class Node:
    __slots__ = {"request_id", "prev", "next"}
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.prev = None
        self.next = None
        
class OrderedSet:
    def __init__(self):
        self.map = {}
        self.potential_list = []
        self.potential_list_index = {}
        self.head = Node(None)  # sentinel node
        self.tail = Node(None)  # sentinel node
        self.head.next = self.tail
        self.tail.prev = self.head

    def add(self, request_id: str, is_potential: bool = False):
        if request_id not in self.map:
            new_node = Node(request_id)
            last = self.tail.prev
            last.next = new_node
            new_node.prev = last
            new_node.next = self.tail
            self.tail.prev = new_node
            self.map[request_id] = new_node
            if is_potential:
                self.potential_list.append(new_node)
                self.potential_list_index[request_id] = len(self.potential_list) - 1

    def discard(self, request_id: str):
        if request_id in self.map:
            node = self.map.pop(request_id)
            node.prev.next = node.next
            node.next.prev = node.prev
            if request_id in self.potential_list_index:
                index = self.potential_list_index.pop(request_id)
                last_node = self.potential_list[-1]
                self.potential_list[index] = last_node
                self.potential_list_index[last_node.request_id] = index
                self.potential_list.pop()

    def __contains__(self, request_id: str) -> bool:
        return request_id in self.map

    def __iter__(self):
        current = self.head.next
        while current != self.tail:
            yield current.request_id
            current = current.next
            
    def find_first_potential_candidate_sample(self, is_official_candidate_sample: dict) -> str | None:
        while self.potential_list:
            candidate_node = self.potential_list[0]
            if is_official_candidate_sample.get(candidate_node.request_id, False):
                # Remove from potential list
                self.potential_list_index.pop(candidate_node.request_id)
                self.potential_list.pop(0)
                return candidate_node.request_id
            else:
                # Not an official candidate anymore, remove it
                self.potential_list_index.pop(candidate_node.request_id)
                self.potential_list.pop(0)
        return None
    
    def clearall(self):
        self.map.clear()
        iter = self.head.next
        while iter != self.tail:
            temp = iter
            iter = iter.next
            del temp
        self.head.next = self.tail
        self.tail.prev = self.head
        