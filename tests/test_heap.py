import unittest
import sys
import os

# Add the parent directory to the path so we can import the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Heap, TopK, BottomK


class TestHeaps(unittest.TestCase):
    """Test cases for the basic Heap class."""

    def test_min_heap_basic(self):
        """Test basic operations of a min-heap."""
        min_heap = Heap[int]()
        min_heap.push(5)
        min_heap.push(1)
        min_heap.push(3)

        self.assertEqual(len(min_heap), 3)
        self.assertEqual(min_heap.peek(), 1)
        self.assertEqual(min_heap.to_sorted_list(), [1, 3, 5])
        self.assertEqual(min_heap.pop(), 1)
        self.assertEqual(len(min_heap), 2)
        self.assertEqual(min_heap.peek(), 3)
        self.assertEqual(min_heap.pop(), 3)
        self.assertEqual(min_heap.pop(), 5)
        self.assertEqual(len(min_heap), 0)

        with self.assertRaises(IndexError):
            min_heap.peek()

        with self.assertRaises(IndexError):
            min_heap.pop()

    def test_max_heap_basic(self):
        """Test basic operations of a max-heap."""
        max_heap = Heap[int](max_heap=True)
        max_heap.push(5)
        max_heap.push(1)
        max_heap.push(8)
        max_heap.push(3)

        self.assertEqual(len(max_heap), 4)
        self.assertEqual(max_heap.peek(), 8)
        self.assertEqual(max_heap.to_sorted_list(), [8, 5, 3, 1])
        self.assertEqual(max_heap.pop(), 8)
        self.assertEqual(len(max_heap), 3)
        self.assertEqual(max_heap.peek(), 5)
        self.assertEqual(max_heap.pop(), 5)
        self.assertEqual(max_heap.pop(), 3)
        self.assertEqual(max_heap.pop(), 1)
        self.assertEqual(len(max_heap), 0)

    def test_representation(self):
        """Test the string representation of the heap."""
        min_heap = Heap[int]()
        min_heap.push(5)
        min_heap.push(1)
        min_heap.push(3)

        # Check that the __repr__ method returns a string that includes the type and content
        repr_str = repr(min_heap)
        self.assertIn("MinHeap", repr_str)
        self.assertIn("[1, 3, 5]", repr_str)
        self.assertNotIn("max_size", repr_str)  # Ensure max_size is gone
        self.assertNotIn("unlimited", repr_str)

        max_heap = Heap[int](max_heap=True)
        max_heap.push(5)
        max_heap.push(1)
        max_heap.push(3)

        # Check that the __repr__ method returns a string that includes the type and content
        repr_str = repr(max_heap)
        self.assertIn("MaxHeap", repr_str)
        self.assertIn("[5, 3, 1]", repr_str)
        self.assertNotIn("max_size", repr_str)  # Ensure max_size is gone

    def test_iteration(self):
        """Test iteration over the heap."""
        min_heap = Heap[int]()
        min_heap.push(5)
        min_heap.push(1)
        min_heap.push(3)

        # Check that iteration returns all items in the heap
        items = list(min_heap)
        self.assertEqual(len(items), 3)
        self.assertSetEqual(set(items), {1, 3, 5})

        # Check that iteration doesn't change the heap
        self.assertEqual(len(min_heap), 3)
        self.assertEqual(min_heap.peek(), 1)


class TestTopK(unittest.TestCase):
    """Test cases for the TopK class."""

    def test_top_k_basic(self):
        """Test basic TopK operations (keeps k largest)."""
        top_3 = TopK[int](k=3)
        top_3.push(10)
        top_3.push(2)
        top_3.push(8)

        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3.peek(), 2)  # Smallest of the top 3
        self.assertCountEqual(top_3.to_list(sorted_list=False), [10, 2, 8])
        self.assertEqual(top_3.to_list(sorted_list=True), [10, 8, 2])

        # 5 > peek (2), so pushpop occurs: push 5, pop 2
        top_3.push(5)
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3.peek(), 5)  # New smallest of top 3
        self.assertEqual(top_3.to_list(), [10, 8, 5])

        # 1 < peek (5), so nothing happens
        top_3.push(1)
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3.peek(), 5)
        self.assertEqual(top_3.to_list(), [10, 8, 5])

        # 12 > peek (5), so pushpop occurs: push 12, pop 5
        top_3.push(12)
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3.peek(), 8)  # New smallest of top 3
        self.assertEqual(top_3.to_list(), [12, 10, 8])

        # Test pop
        self.assertEqual(top_3.pop(), 8)  # Removes smallest of top 3
        self.assertEqual(len(top_3), 2)
        self.assertEqual(top_3.peek(), 10)
        self.assertEqual(top_3.to_list(), [12, 10])

    def test_top_k_less_than_k_elements(self):
        """Test TopK when fewer than K elements are pushed."""
        top_5 = TopK[int](k=5)
        top_5.push(10)
        top_5.push(2)
        self.assertEqual(len(top_5), 2)
        self.assertEqual(top_5.peek(), 2)
        self.assertEqual(top_5.to_list(), [10, 2])
        self.assertEqual(top_5.pop(), 2)
        self.assertEqual(len(top_5), 1)
        self.assertEqual(top_5.peek(), 10)
        self.assertEqual(top_5.pop(), 10)
        self.assertEqual(len(top_5), 0)
        with self.assertRaises(IndexError):
            top_5.peek()
        with self.assertRaises(IndexError):
            top_5.pop()

    def test_top_k_iteration(self):
        """Test iteration over TopK."""
        top_3 = TopK[int](k=3)
        top_3.push(10)
        top_3.push(2)
        top_3.push(8)
        top_3.push(5)  # replaces 2
        items = list(top_3)
        self.assertEqual(len(items), 3)
        self.assertSetEqual(set(items), {10, 8, 5})
        # Check that iteration doesn't change the tracker
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3.peek(), 5)

    def test_top_k_representation(self):
        """Test the string representation of TopK."""
        top_3 = TopK[int](k=3)
        top_3.push(10)
        top_3.push(2)
        top_3.push(5)
        repr_str = repr(top_3)
        self.assertIn("TopK", repr_str)
        self.assertIn("k=3", repr_str)
        self.assertIn("content=[10, 5, 2]", repr_str)  # Sorted descending

    def test_top_k_invalid_k(self):
        """Test initializing TopK with invalid k."""
        with self.assertRaises(ValueError):
            TopK[int](k=0)
        with self.assertRaises(ValueError):
            TopK[int](k=-1)


class TestBottomK(unittest.TestCase):
    """Test cases for the BottomK class."""

    def test_bottom_k_basic(self):
        """Test basic BottomK operations (keeps k smallest)."""
        bottom_3 = BottomK[int](k=3)
        bottom_3.push(10)
        bottom_3.push(2)
        bottom_3.push(8)

        self.assertEqual(len(bottom_3), 3)
        self.assertEqual(bottom_3.peek(), 10)  # Largest of the bottom 3
        self.assertCountEqual(bottom_3.to_list(sorted_list=False), [10, 2, 8])
        self.assertEqual(bottom_3.to_list(sorted_list=True), [2, 8, 10])

        # 5 < peek (10), so pushpop occurs: push 5, pop 10
        bottom_3.push(5)
        self.assertEqual(len(bottom_3), 3)
        self.assertEqual(bottom_3.peek(), 8)  # New largest of bottom 3
        self.assertEqual(bottom_3.to_list(), [2, 5, 8])

        # 12 > peek (8), so nothing happens
        bottom_3.push(12)
        self.assertEqual(len(bottom_3), 3)
        self.assertEqual(bottom_3.peek(), 8)
        self.assertEqual(bottom_3.to_list(), [2, 5, 8])

        # 1 < peek (8), so pushpop occurs: push 1, pop 8
        bottom_3.push(1)
        self.assertEqual(len(bottom_3), 3)
        self.assertEqual(bottom_3.peek(), 5)  # New largest of bottom 3
        self.assertEqual(bottom_3.to_list(), [1, 2, 5])

        # Test pop
        self.assertEqual(bottom_3.pop(), 5)  # Removes largest of bottom 3
        self.assertEqual(len(bottom_3), 2)
        self.assertEqual(bottom_3.peek(), 2)
        self.assertEqual(bottom_3.to_list(), [1, 2])

    def test_bottom_k_less_than_k_elements(self):
        """Test BottomK when fewer than K elements are pushed."""
        bottom_5 = BottomK[int](k=5)
        bottom_5.push(10)
        bottom_5.push(2)
        self.assertEqual(len(bottom_5), 2)
        self.assertEqual(bottom_5.peek(), 10)
        self.assertEqual(bottom_5.to_list(), [2, 10])
        self.assertEqual(bottom_5.pop(), 10)
        self.assertEqual(len(bottom_5), 1)
        self.assertEqual(bottom_5.peek(), 2)
        self.assertEqual(bottom_5.pop(), 2)
        self.assertEqual(len(bottom_5), 0)
        with self.assertRaises(IndexError):
            bottom_5.peek()
        with self.assertRaises(IndexError):
            bottom_5.pop()

    def test_bottom_k_iteration(self):
        """Test iteration over BottomK."""
        bottom_3 = BottomK[int](k=3)
        bottom_3.push(10)
        bottom_3.push(2)
        bottom_3.push(8)
        bottom_3.push(5)  # replaces 10
        items = list(bottom_3)  # Iteration yields actual values
        self.assertEqual(len(items), 3)
        self.assertSetEqual(set(items), {2, 8, 5})
        # Check that iteration doesn't change the tracker
        self.assertEqual(len(bottom_3), 3)
        self.assertEqual(bottom_3.peek(), 8)

    def test_bottom_k_representation(self):
        """Test the string representation of BottomK."""
        bottom_3 = BottomK[int](k=3)
        bottom_3.push(10)
        bottom_3.push(2)
        bottom_3.push(5)
        repr_str = repr(bottom_3)
        self.assertIn("BottomK", repr_str)
        self.assertIn("k=3", repr_str)
        self.assertIn("content=[2, 5, 10]", repr_str)  # Sorted ascending

    def test_bottom_k_invalid_k(self):
        """Test initializing BottomK with invalid k."""
        with self.assertRaises(ValueError):
            BottomK[int](k=0)
        with self.assertRaises(ValueError):
            BottomK[int](k=-1)


if __name__ == "__main__":
    unittest.main()
