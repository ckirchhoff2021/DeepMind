import copy
from unittest import TestCase
from collections import OrderedDict

def longest_subarray_1493(nums):
    N = len(nums)
    dp = [0] * N
    dp[0] = nums[0]
    for i in range(1, N):
        if nums[i] == 0:
            dp[i] = 0
        else:
            dp[i] = dp[i-1] + 1
    left = 0
    ret = 0
    contain_zero = False
    for i in range(N):
        if dp[i] == 0 and i > 0:
            cur = left + dp[i-1]
            ret = cur if cur > ret else ret
            left = dp[i-1]
            contain_zero = True
        else:
            if i == N -1:
                cur = left + dp[i]
                ret = cur if cur > ret else ret
    ret = ret if contain_zero else ret - 1
    return ret


def simplify_path_71(path):
    values = path.split('/')
    ret = list()
    for val in values:
        if len(val) == 0:
            continue
        if val == '.':
            continue
        elif val == '..':
            if len(ret) > 0:
                ret.pop()
        else:
            ret.append(val)
    return '/' + '/'.join(ret)


def n_queen(n):
    ret = list()

    def check_valid(path):
        if len(path) == n:
            ret.append(path[:])
            return
        row = len(path)
        for col in range(n):
            valid = True
            for pos in path:
                i, j = pos
                if row == i or col == j or abs(i - row) == abs(j - col):
                    valid = False
                    break
            if not valid:
                continue
            path.append((row, col))
            check_valid(path)
            path.pop()
    paths = list()
    check_valid(paths)
    return ret


def print_n_queen(x):
    n = len(x)
    zeros = [[0] * n for i in range(n)]
    for (i, j) in x:
        zeros[i][j] = 1
    for i in range(n):
        print(' '.join([str(val) for val in zeros[i]]))
    print('\n')


def total_n_queens(n):
    ret = n_queen(n)
    return len(ret)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    @staticmethod
    def create_list(values):
        if len(values) == 0:
            return None
        head = ListNode(values[0])
        p = head
        for i in range(1, len(values)):
            p.next = ListNode(values[i])
            p = p.next
        return head

    def get_data(self):
        ret = [self.val]
        p = self.next
        while p:
            ret.append(p.val)
            p = p.next
        return ret


def delete_duplicates_83(head):
    root = head
    if root is None or root.next is None:
        return root
    prev = root
    prev.delete = False
    val = prev.val
    cur = root.next
    while cur:
        if cur.val == val:
            cur.delete = True
            prev.delete = True
        else:
            cur.delete = False
        val = cur.val
        prev = cur
        cur = cur.next

    p = head
    while p and p.delete:
        p = p.next
    if p is None or p.next is None:
        return p
    prev = p
    q = p.next
    while q:
        if q.delete:
            q = q.next
            prev.next = q
        else:
            prev = q
            q = q.next
    return root


def min_window_76(s, t):
    m = len(s)
    n = len(t)
    if m < n:
        return ""
    start = -1
    end = m - 1
    for i in range(0, m-n+1):
        if s[i] not in t:
            continue
        target = list(t)
        for j in range(i, m):
            if s[j] in target:
                target.remove(s[j])
            if len(target) == 0:
                if j - i < end - start:
                    start, end = i, j
                break
    if start == -1:
        return ""
    return s[start:end+1]


def min_window_speed_76(s, t):
    m = len(s)
    n = len(t)
    if m < n:
        return ""

    indices = []
    for i in range(m):
        if s[i] in t:
            indices.append(i)
    
    p = list(t)
    for i in range(n):
        if s[i] in t:
            p.remove(s[i])

    for i in range(0, m-n+1):
        s1 = s[i:i+n]

    start = -1
    end = m - 1
    for i in range(0, m-n+1):
        if s[i] not in t:
            continue
        target = list(t)
        for j in range(i, m):
            if s[j] in target:
                target.remove(s[j])
            if len(target) == 0:
                if j - i < end - start:
                    start, end = i, j
                break
    if start == -1:
        return ""
    return s[start:end+1]


def is_scramble_step(x1, x2, d1):
    if x1 == x2:
        return True
    k1 = OrderedDict()
    k2 = OrderedDict()
    for x in d1:
        k1[x] = 0
        k2[x] = 0
    N = len(x1)
    for i in range(N):
        a, b = x1[i], x2[i]
        k1[a] += 1
        k2[b] += 1
        if str(k1) != str(k2):
            continue
        if i == N - 1:
            return True
        b1 = is_scramble_step(x1[:i+1], x2[:i+1], k1)
        b2 = is_scramble_step(x1[i+1:], x2[i+1:], k1)
        if b1 and b2:
            return True

    k3 = OrderedDict()
    k4 = OrderedDict()
    for x in d1:
        k3[x] = 0
        k4[x] = 0
    for i in range(N):
        a, b = x1[i], x2[N-1-i]
        k3[a] += 1
        k4[b] += 1
        if str(k3) != str(k4):
            continue
        if i == N-1:
            return True
        b1 = is_scramble_step(x1[:i + 1], x2[N-1-i:], k3)
        b2 = is_scramble_step(x1[i + 1:], x2[:N-1-i], k3)
        if b1 and b2:
            return True
    return False


def is_scramble(x1, x2):
    if x1 == x2:
        return True
    if len(x1) != len(x2):
        return False
    if len(x1) == 1:
        return x1 == x2
    N = len(x1)
    d1 = dict()
    d2 = dict()
    for i in range(N):
        a, b = x1[i], x2[i]
        d1[a] = d1.get(a, 0) + 1
        d2[b] = d2.get(b, 0) + 1
    for x in d1:
        if x not in d2 or d1[x] != d2[x]:
            return False
    ret = is_scramble_step(x1, x2, d1)
    return ret


class TestDailyCode(TestCase):
    def test_longest_subarray(self):
        nums = [1, 1, 1]
        ret = longest_subarray_1493(nums)
        self.assertEqual(ret, 2)

    def test_simplify_path(self):
        path = "/../"
        expect = "/"
        ret = simplify_path_71(path)
        self.assertEqual(expect, ret)

    def test_n_queen(self):
        ret = n_queen(8)
        print(ret)
        print(len(ret))
        expect = True
        for x in ret:
            print_n_queen(x)
        self.assertEqual(True, expect)

    def test_delete_duplicates(self):
        head = ListNode.create_list([1,2,3,3,4,4,5])
        head = delete_duplicates_83(head)
        ret = head.get_data()
        print(ret)
        expect = (1,2,5)
        self.assertEqual(expect, tuple(ret))

    def test_min_window(self):
        s = "ADOBECODEBANC"
        t = "ABC"
        ret = min_window_76(s, t)
        expect = "BANC"
        self.assertEqual(ret, expect)

    def test_scramble(self):
        s1 = "great"
        s2 = "rgeat"
        ret = is_scramble(s1, s2)
        self.assertTrue(ret)