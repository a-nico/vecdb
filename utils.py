import heapq
from typing import TypeVar, Generic, Optional, List, Union, Iterable

T = TypeVar("T")


class Heap(Generic[T]):
    """
    A generic heap implementation using Python's heapq module.
    Can be configured as a min-heap or max-heap.
    """

    def __init__(self, max_heap: bool = False):
        """
        Initializes the Heap.

        Args:
            max_heap (bool): If True, creates a max-heap. Otherwise, creates a min-heap (default).
        """
        self._heap: List[Union[T, float]] = []  # Internal list to store heap elements
        self._max_heap = max_heap
        # For max-heap, we store negated values to use heapq's min-heap implementation
        self._multiplier = -1 if max_heap else 1

    def push(self, item: T) -> None:
        """
        Adds an item to the heap.

        Args:
            item (T): The item to add to the heap.
        """
        # Negate item if it's a max-heap
        heap_item = self._multiplier * item
        heapq.heappush(self._heap, heap_item)

    def peek(self) -> T:
        """
        Returns the smallest (for min-heap) or largest (for max-heap)
        item from the heap without removing it.

        Returns:
            T: The top item.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._heap:
            raise IndexError("peek from empty heap")
        # Return the actual value (de-negate if max-heap)
        return self._multiplier * self._heap[0]

    def pop(self) -> T:
        """
        Removes and returns the smallest (for min-heap) or largest
        (for max-heap) item from the heap.

        Returns:
            T: The removed top item.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty heap")
        # Pop the item
        popped_item = heapq.heappop(self._heap)
        # Return the actual value (de-negate if max-heap)
        return self._multiplier * popped_item

    def __len__(self) -> int:
        """Returns the current number of items in the heap."""
        return len(self._heap)

    def __iter__(self):
        """Allows iteration over the heap items (not necessarily sorted)."""
        # Note: Iterating directly over self._heap gives internal values (negated for max-heap)
        # This iterator yields the actual values.
        return (self._multiplier * item for item in self._heap)

    def __repr__(self) -> str:
        """Returns a string representation of the heap."""
        heap_type = "MaxHeap" if self._max_heap else "MinHeap"
        # Convert internal representation back to actual values for display
        content = sorted(
            [self._multiplier * item for item in self._heap], reverse=self._max_heap
        )
        return f"{heap_type}({content})"

    def to_sorted_list(self) -> List[T]:
        """
        Returns a new list containing all elements from the heap in sorted order
        (ascending for min-heap, descending for max-heap) without modifying the heap.

        Returns:
            List[T]: A sorted list of all elements in the heap.
        """
        # Convert internal representation back to actual values and sort appropriately
        return sorted(
            [self._multiplier * item for item in self._heap], reverse=self._max_heap
        )


class TopK(Generic[T]):
    """
    Keeps track of the K largest elements seen so far.
    Uses a min-heap internally.
    """

    def __init__(self, k: int):
        """
        Initializes the TopK tracker.

        Args:
            k (int): The number of largest elements to keep track of.
                     Must be greater than 0.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self._k = k
        self._heap: List[Union[T, float]] = []  # Min-heap

    def push(self, item: T) -> None:
        """
        Adds an item to the tracker.

        If the tracker holds less than K items, the item is added.
        If the tracker is full (holds K items), the new item is added
        only if it is larger than the smallest item currently in the tracker.
        If added when full, the smallest item is removed.

        Args:
            item (T): The item to potentially add.
        """
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:  # Compare with the smallest item in the min-heap
            heapq.heappushpop(self._heap, item)

    def peek(self) -> T:
        """
        Returns the smallest item among the top K elements currently held.

        Returns:
            T: The smallest of the top K items.

        Raises:
            IndexError: If the tracker is empty.
        """
        if not self._heap:
            raise IndexError("peek from empty TopK tracker")
        return self._heap[0]

    def pop(self) -> T:
        """
        Removes and returns the smallest item among the top K elements.
        This effectively reduces the number of items tracked if called before K items are pushed.
        It's generally used after K items have been pushed to extract the Kth largest element.

        Returns:
            T: The removed smallest item from the top K.

        Raises:
            IndexError: If the tracker is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty TopK tracker")
        # Note: heapq.heappop removes the smallest item from the min-heap
        return heapq.heappop(self._heap)

    def to_list(self, sorted_list: bool = True) -> List[T]:
        """
        Returns a list of the top K elements currently held.

        Args:
            sorted_list (bool): If True (default), returns the list sorted in descending order.
                                Otherwise, the order is arbitrary.

        Returns:
            List[T]: A list of the elements.
        """
        if sorted_list:
            return sorted(self._heap, reverse=True)
        else:
            return list(self._heap)

    def __len__(self) -> int:
        """Returns the current number of items held (at most K)."""
        return len(self._heap)

    def __iter__(self) -> Iterable[T]:
        """Allows iteration over the items currently held (arbitrary order)."""
        return iter(self._heap)

    def __repr__(self) -> str:
        """Returns a string representation of the TopK tracker."""
        content = sorted(self._heap, reverse=True)
        return f"TopK(k={self._k}, content={content})"


class BottomK(Generic[T]):
    """
    Keeps track of the K smallest elements seen so far.
    Uses a max-heap internally.
    """

    def __init__(self, k: int):
        """
        Initializes the BottomK tracker.

        Args:
            k (int): The number of smallest elements to keep track of.
                     Must be greater than 0.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self._k = k
        # Max-heap: store negated values
        self._heap: List[Union[T, float]] = []

    def _negate(self, item):
        # If item is a tuple, negate only the first element
        if isinstance(item, tuple):
            return (-item[0], *item[1:])
        else:
            return -item

    def _denegate(self, item):
        # If item is a tuple, de-negate only the first element
        if isinstance(item, tuple):
            return (-item[0], *item[1:])
        else:
            return -item

    def push(self, item: T) -> None:
        """
        Adds an item to the tracker.

        If the tracker holds less than K items, the item is added.
        If the tracker is full (holds K items), the new item is added
        only if it is smaller than the largest item currently in the tracker.
        If added when full, the largest item is removed.

        Args:
            item (T): The item to potentially add.
        """
        negated_item = self._negate(item)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, negated_item)
        # Compare negated_item with the smallest negated value (largest actual value)
        elif negated_item > self._heap[0]:
            heapq.heappushpop(self._heap, negated_item)

    def peek(self) -> T:
        """
        Returns the largest item among the bottom K elements currently held.

        Returns:
            T: The largest of the bottom K items.

        Raises:
            IndexError: If the tracker is empty.
        """
        if not self._heap:
            raise IndexError("peek from empty BottomK tracker")
        # Return the actual value (de-negate)
        return self._denegate(self._heap[0])

    def pop(self) -> T:
        """
        Removes and returns the largest item among the bottom K elements.
        This effectively reduces the number of items tracked if called before K items are pushed.
        It's generally used after K items have been pushed to extract the Kth smallest element.

        Returns:
            T: The removed largest item from the bottom K.

        Raises:
            IndexError: If the tracker is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty BottomK tracker")
        # Pop the smallest negated item (largest actual item)
        popped_negated_item = heapq.heappop(self._heap)
        # Return the actual value (de-negate)
        return self._denegate(popped_negated_item)

    def to_list(self, sorted_list: bool = True) -> List[T]:
        """
        Returns a list of the bottom K elements currently held.

        Args:
            sorted_list (bool): If True (default), returns the list sorted in ascending order.
                                Otherwise, the order is arbitrary.

        Returns:
            List[T]: A list of the elements.
        """
        # De-negate items before returning/sorting
        actual_items = [self._denegate(item) for item in self._heap]
        if sorted_list:
            return sorted(actual_items)
        else:
            return actual_items

    def __len__(self) -> int:
        """Returns the current number of items held (at most K)."""
        return len(self._heap)

    def __iter__(self) -> Iterable[T]:
        """Allows iteration over the items currently held (actual values, arbitrary order)."""
        return (self._denegate(item) for item in self._heap)

    def __repr__(self) -> str:
        """Returns a string representation of the BottomK tracker."""
        content = sorted([self._denegate(item) for item in self._heap])
        return f"BottomK(k={self._k}, content={content})"
