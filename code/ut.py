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
