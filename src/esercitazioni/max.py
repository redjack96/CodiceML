from typing import List, Tuple
import unittest

def argmax(list: List[float]) -> int:
    best_i = 0
    max = 0
    for i in range(len(list)):
        val = list[i]
        if max < val:
            max = val
            best_i = i
    return best_i

def nth_argmax(list: List[float], n) -> int:
    copied = list.copy()

    curr = 0 # current maximum nth
    curr_argmax = 0
    while curr < n:
        curr_argmax = argmax(copied)
        copied[curr_argmax] = 0
        curr += 1
    return curr_argmax

def get_top_k(list: List[float], n) -> List[Tuple[int, float]]:
    top_k_list = []
    for i in range(1,n+1):
        nth = nth_argmax(list, i)
        top_k_list.append((nth, list[nth]))
    
    return top_k_list

class MaxTest(unittest.TestCase):
        def testArgMax(self): # test method names begin with 'test'
            self.assertEqual(argmax([0,1,2,3,4]), 4)
            self.assertEqual(argmax([4,3,2,1,0]), 0)
            self.assertEqual(argmax([4.1,3.2,2.3,1.4,0.1]), 0)
        
        def testNthArgMax(self):
            self.assertEqual(nth_argmax([0,1,2,3,4], 3), 2)
            self.assertEqual(nth_argmax([4,3,2,1,0], 1), 0)
            self.assertEqual(nth_argmax([4.1,3.1,2.1,1.1,0.1], 1), 0)
            self.assertEqual(nth_argmax([4.1,3.1,2.1,1.1,0.1], 2), 1)

        def testTopK(self):
            self.assertEqual(get_top_k([0,3,1,6,2,5], 3), [(3, 6),(5, 5), (1,3)])

if __name__ == '__main__':
    unittest.main()