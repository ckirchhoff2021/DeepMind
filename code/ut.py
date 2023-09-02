import copy
from unittest import TestCase
from collections import OrderedDict
from .base import TreeNode

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


def isBalanced(root):
    def get_depth(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            node.depth = 1
            return 1
        left = get_depth(node.left)
        right = get_depth(node.right)
        depth = max(left, right) + 1
        node.depth = depth
        return depth

    if root is None:
        return True
    get_depth(root)
        
    def checkBalanced(node):
        if node is None:
            return True
        left = node.left
        right = node.right
        if left is None and right is None:
            return True
        if left is None and right.depth > 1:
            return False
        if right is None and left.depth > 1:
            return False
        if left and right and abs(left.depth - right.depth) > 1:
            return False
        return checkBalanced(left) and checkBalanced(right)
    return checkBalanced(root)


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


def pathSum(root, targetSum):
    ret = list()
    if root is None:
        return ret
    def traverse(node, target, path):
        val = node.val
        remain = target - val
        path.append(node.val)
        if remain == 0 and node.left is None and node.right is None:
            ret.append(path[:])
        if node.left:
            traverse(node.left, remain, path)
            path.pop()
        if node.right:
            traverse(node.right, remain, path)
            path.pop()
    traverse(root, targetSum, [])
    return ret


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


def flatten(root):
    def visit(node, values, nodes):
        if node is None:
            return
        values.append(node.val)
        nodes.append(node)
        visit(node.left, values, nodes)
        visit(node.right, values, nodes)
    values = []
    nodes = []
    visit(root, values, nodes)
    N = len(values)
    for i in range(N):
        node = nodes[N-1-i]
        node.val = values[N-1-i]
        node.left = None
        if i == 0:
            node.right = None
        else:
            node.right = nodes[N-i]
    return root


def build_tree_pre_in(pre_order, in_order):
    if len(pre_order) == 0:
        return None
    root = TreeNode(pre_order[0])
    index = in_order.index(pre_order[0])
    left_in = in_order[:index]
    right_in = in_order[index+1:]
    left_num = len(left_in)
    left_pre = pre_order[1:left_num+1]
    right_pre = pre_order[left_num+1:]
    left = build_tree_pre_in(left_pre, left_in)
    right = build_tree_pre_in(right_pre, right_in)
    root.left = left
    root.right = right
    return root


def build_tree_in_post(inorder, postorder):
    if len(inorder) == 0:
        return None
    root = TreeNode(postorder[-1])
    index = inorder.index(postorder[-1])
    left_in = inorder[:index]
    right_in = inorder[index+1:]
    left_num = len(left_in)
    left_post = postorder[:left_num]
    right_post = postorder[left_num:-1]
    left = build_tree_in_post(left_in, left_post)
    right = build_tree_in_post(right_in, right_post)
    root.left = left
    root.right = right
    return root


def max_profit_122(stocks):
    N = len(stocks)
    ret = 0
    for i in range(1, N):
        profit = stocks[i] - stocks[i-1]
        if profit < 0:
            profit = 0
        ret += profit
    return ret


def sorted_array_t_BST_108(nums):
    N = len(nums)
    if N == 0:
        return None
    mid = N // 2
    left_values = nums[:mid]
    root = TreeNode(nums[mid])
    left = sorted_array_t_BST_108(left_values)
    right_values = nums[mid+1:]
    right = sorted_array_t_BST_108(right_values)
    root.left = left
    root.right = right
    return root


def longest_consecutive_128(nums):
    values = sorted(nums)
    N = len(values)
    dp = [1] * N
    for i in range(1, N):
        if values[i] == values[i-1] + 1:
            dp[i] = dp[i-1] + 1
        if values[i] == values[i-1]:
            dp[i] = dp[i-1]
    return max(dp)


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

    def test_path_sum(self):
        values = [5,4,8,11,None,13,4,7,2,None,None,5,1]
        root = TreeNode.create(values)
        ret = pathSum(root, 22)
        output = [[5,4,11,2],[5,8,4,5]]
        self.assertIn(output[0], ret)
        self.assertIn(output[1], ret)

    def test_binary_balanced(self):
        values = [3,9,20,None,None,15,7]
        root = TreeNode.create(values)
        ret = isBalanced(root)
        self.assertTrue(ret)

    def test_build_tree_pre_in(self):
        pre_order = [6, 5, 2, 3, 1, 4, 2, 8]
        in_order = [2, 5, 1, 3, 6, 2, 4, 8]
        root = build_tree_pre_in(pre_order, in_order)
        value = root.left.right.left.val
        self.assertEqual(value, 1)

    def test_build_tree_in_post(self):
        post_order = [2, 1, 3, 5, 2, 8, 4, 6]
        in_order = [2, 5, 1, 3, 6, 2, 4, 8]
        root = build_tree_in_post(in_order, post_order)
        value = root.left.right.left.val
        self.assertEqual(value, 1)

    def test_max_profit122(self):
        stocks = [7, 1, 5, 3, 6, 4]
        profit = max_profit_122(stocks)
        ret = 7
        self.assertEqual(profit, ret)

    def test_sorted_array_t_BST(self):
        values = [-10, -3, 0, 5, 9]
        root = sorted_array_t_BST_108(values)
        ret = root.right.val
        self.assertEqual(ret, 9)

    def test_longest_consecutive(self):
        # values = [100, 4, 200, 1, 3, 2]
        values = [1, 2, 0, 1]
        num = longest_consecutive_128(values)
        self.assertEqual(num, 3)

