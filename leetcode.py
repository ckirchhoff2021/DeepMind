class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        def get_list_value(root):
            node = root
            values = []
            while node:
                values.append(node.val)
                node = node.next
            ret = 0
            depth = len(values)
            for i in range(depth):
                ret += values[depth-1- i] * (10 ** (depth-1- i))
            return ret
        
        k1 = get_list_value(l1)
        k2 = get_list_value(l2)
        # print('k1, k2: ', k1, k2)
        values = str(k1 + k2)
        count = len(values)
        root = ListNode()
        node = root
        for i in range(count):
            val = values[count-1-i]
            node.val = int(val)
            if i < count - 1:
                node.next = ListNode()
            node = node.next
        return root

    def threeSum(self, nums):
        arr = sorted(nums)
        count = len(arr)
        ret = list()
        for i in range(count):
            v1 = arr[i]
            for j in range(i+1,count):
                data = arr[j+1:]
                v2 = arr[j]
                v3 = (v1 + v2) * (-1)
                if v3 in data:
                    seq = [v1, v2, v3]
                    if seq not in ret:
                        ret.append(seq)
        result = [list(val) for val in ret]
        return result

    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        count = len(nums)
        k1 = -1
        k2 = -1
        tag = False
        for i in range(count-1, -1, -1):
            a1 = nums[i]
            for j in range(i-1, -1, -1):
                a2 = nums[j]
                if a1 > a2:
                    print(i, j)
                    k1 = i
                    k2 = j
                    tag = True
                    break
            if tag:
                break
        if not tag:
            nums = list(reversed(nums))
        else:
            value = nums[k1]
            nums[k1] = nums[k2]
            nums[k2] = value

    def longestPalindrome(self, s):
        count = len(s)
        dp = list()
        for i in range(count):
            data = [False] * count
            dp.append(data)

        for i in range(count):
            for j in range(count):
                if i >= j:
                    dp[i][j] = True
        
        k1 = k2 = max_len= 0
        for j in range(1, count):
            for i in range(j):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1]
                    if dp[i][j] and j - i + 1 > max_len:
                        k1 = i
                        k2 = j
                        max_len = j -i +1
                        print(max_len)
                else:
                    dp[i][j] = False

        return s[k1:k2+1]

    def numDecodings(self, s):
        if len(s) == 0 or s[0] == '0':
            return 0

        dp = [1] * len(s)
        for i in range(1, len(s)):
            k1 = int(s[i-1])
            k2 = int(s[i])
            if k1 == 0 and k2 == 0:
                return 0
            elif k1 == 0 and k2 != 0:
                dp[i] = dp[i-1]
            elif k1 != 0 and k2 == 0:
                if k1 > 2:
                    return 0
                else:
                    if i >= 2:
                        dp[i] = dp[i-2]
                    else:
                        dp[i] = 1
            else:
                if k1 > 2 or (k1 == 2 and k2 > 6):
                    dp[i] = dp[i-1]
                else:
                    if i >= 2:
                        dp[i] = dp[i-1] + dp[i-2]
                    else:
                        dp[i] = dp[i-1] + 1

        return dp[-1]

    def countAndSay(self, n) -> str:
        if n == 1:
            return '1'
        values = self.countAndSay(n-1)
        count = len(values)
        if count == 1:
            return '1' + values[0]
        ret = ''
        i = 1
        k = 1
        while i < count:
            if values[i] == values[i-1]:
                k += 1
                if i == count - 1:
                    ret += (str(k) + values[i])
            else:
                ret += (str(k) + values[i-1])
                if i == count -1:
                    ret += ('1'+values[i])
                k = 1
            i += 1
        return ret                
            


             
def merged_sorted(a1, a2):
    i1 = 0
    i2 = 0
    a3 = list()
    while i1 < len(a1) and i2 < len(a2):
        if a1[i1] < a2[i2]:
            a3.append(a1[i1])
            i1 += 1
        else:
            a3.append(a2[i2])
            i2 += 1
    
    while i1 < len(a1):
        a3.append(a1[i1])
        i1 += 1
    
    while i2 < len(a2):
        a3.append(a2[i2])
        i2 += 1
    return a3



def main():
    solution = Solution()
    print("-- addTwoNumbers --")
    k1 = ListNode(5)
    k1.next = ListNode(6)
    k1.next.next = ListNode(4)
    k1.next.next.next = ListNode(9)

    k2 = ListNode(2)
    k2.next = ListNode(4)
    k2.next.next = ListNode(9)

    k3 = solution.addTwoNumbers(k1, k2)
    node = k3
    while node:
        print(node.val)
        node = node.next

    print("-- threeSum --")
    nums = [-1,0,1,2,-1,4]
    arr = solution.threeSum(nums)
    print(arr)

    print("-- nextPermutation --")
    nums = [3,2,1]
    solution.nextPermutation(nums)
    print(nums)

    print('-- longestPalindrome -- ')
    str_data = 'abcbad'
    value = solution.longestPalindrome(str_data)
    print(value)

    print('-- num decoding --')
    str_data = '112352'
    value = solution.numDecodings(str_data)
    print(value)


    print('-- count and say --')
    value = solution.countAndSay(6)
    print(value)

if __name__ == '__main__':
    main()

    # x = [1,3,5]
    # y = [2,4]
    # z = merged_sorted(x, y)
    # print(z)

        