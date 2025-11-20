class Node:
    __slots__ = {"request_id", "prev", "next"}
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.prev = None
        self.next = None
        
class OrderedSet:
    def __init__(self):
        self.map = {}
        self.head = Node(None)  # sentinel node
        self.tail = Node(None)  # sentinel node
        self.head.next = self.tail
        self.tail.prev = self.head

    def add(self, request_id: str):
        if request_id not in self.map:
            new_node = Node(request_id)
            last = self.tail.prev
            last.next = new_node
            new_node.prev = last
            new_node.next = self.tail
            self.tail.prev = new_node
            self.map[request_id] = new_node

    def discard(self, request_id: str):
        if request_id in self.map:
            node = self.map.pop(request_id)
            node.prev.next = node.next
            node.next.prev = node.prev

    def __contains__(self, request_id: str) -> bool:
        return request_id in self.map

    def __iter__(self):
        current = self.head.next
        while current != self.tail:
            yield current.request_id
            current = current.next
            
    def find_first_potential_candidate_sample(self, is_official_candidate_sample: dict) -> str | None:
        current = self.tail.prev
        while current != self.head:
            if not is_official_candidate_sample.get(current.request_id, True):
                return current.request_id
            current = current.prev
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
        