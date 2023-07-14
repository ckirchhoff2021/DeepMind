from unittest import TestCase


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