import copy
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
        k = count - 1
        while k > 0:
            if nums[k-1] >= nums[k]:
                k = k -1
            else:
                break

        if k == 0:
            reverse = [nums[i] for i in range(count-1,-1,-1)]
            for i in range(count):
                nums[i] = reverse[i]
        else:
            j = count - 1
            while nums[j] <= nums[k-1] and j > k - 1:
                j = j - 1

            if nums[j] > nums[k-1] and j > k -1:
                tmp = nums[k-1]
                nums[k-1] = nums[j]
                nums[j] = tmp

                remains = sorted(nums[k:])
                for s in range(k,count):
                    nums[s] = remains[s-k]

            '''
            i = k
            while nums[i] <= nums[-1] and i < count - 1:
                i = i + 1

            if nums[i] > nums[-1] and i < count - 1:
                tmp = nums[-1]
                nums[-1] = nums[i]
                nums[i] = tmp   
            '''



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

    def reverse_list(self, root):
        p = root
        q = p.next
        if q is None:
            return p

        p.next = None
        while q:
            r = q.next
            q.next = p
            p = q
            q = r
        return p


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


    def deleteDuplicates(self, head) :
        node = head
        if node is None:
            return None
        values = list()
        nodes = list()
        while node:
            value = node.val
            if value not in values:
                values.append(value)
                nodes.append(node)
            node = node.next
        p = nodes[0]
        p.next = None
        for i in range(1, len(nodes)):
            q = nodes[i]
            q.next = None
            p.next = q
            p = p.next
        return nodes[0]

    @staticmethod
    def valid(s, k1, words):
        if len(words) == 0:
            return True

        for i in range(len(words)):
            word = words[i]
            wlen = len(word)
            k2 = k1 + wlen
            if k2 > len(s):
                continue
            slice = s[k1:k2]
            if slice == word:
                words_copy = words.copy()
                words_copy.pop(i)
                if Solution.valid(s,k2,words_copy):
                    return True
        return False

    def findSubstring(self, s, words):
        ret = list()
        count = len(s)
        for i in range(count):
            if self.valid(s, i, words):
                ret.append(i)
        return ret

    def compare_node(self, p, q):
        if (not p and q) or (p and not q):
            return False
        if p.val != q.val:
            return False
        if not p and not q:
            return True

        p1 = p.left
        p2 = p.right
        q1 = q.right
        q2 = q.left
        return self.compare_node(p1, q1) and self.compare_node(p2, q2)

    def isSymmetric(self, root):
        left = root.left
        right = root.right
        return self.compare_node(left, right)

    def levelOrderBottom(self, root):
        node_list = []
        ret = []
        if root:
            node_list.append(root)
        while len(node_list) > 0:
            layer = []
            nodes = list()
            for node in node_list:
                layer.append(node.val)
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            node_list = nodes
            ret.append(layer)
        ret.reverse()
        return ret

    def maxProfit(self, prices):
        max_profit = 0
        min_price = 100000
        for value in prices:
            if value < min_price:
                min_price = value
            profit = value - min_price
            if profit > max_profit:
                max_profit = profit
        return max_profit

    def groupAnagrams(self, strs):
        if len(strs) == 0:
            return [[""]]

        values = list()
        for a in strs:
            b = sorted(a)
            c = ''.join(b)
            values.append(c)

        rets = list()
        indices = list()
        counts = len(values)
        for i in range(counts):
            if i in indices:
                continue

            d = values[i]
            pairs = list()
            pairs.append(i)
            indices.append(i)

            for j in range(i+1, counts):
                if j in indices:
                    continue

                if values[j] == d:
                    pairs.append(j)
                    indices.append(j)

            rets.append(pairs)

        res = list()
        for pair in rets:
            values = list()
            for x in pair:
                values.append(strs[x])
            res.append(values)
        return res

    def groupAnagrams2(self, strs):
        hash_map = {}
        for word in strs:
            _word = ''.join(sorted(word))
            if _word not in hash_map:
                hash_map[_word] = [word]
            else:
                hash_map[_word].append(word)
        return list(hash_map.values())


    def fourSum(self, nums, target):
        values = sorted(nums)
        if len(values) < 4:
            return []

        def twoSum(values, start, end, score):
            if start >= end:
                return []

            i = start
            j = end
            ret = set()
            while i < j:
                if values[i] + values[j] == score:
                    ret.add((values[i], values[j]))
                    i = i + 1
                elif values[i] + values[j] < score:
                    i = i + 1
                else:
                    j = j - 1
            return list(ret)


        def threeSum(values, start, end, score):
            if start >= end:
                return []

            res = list()
            for i in range(start, end):
                if i > start and values[i] == values[i-1]:
                    continue

                k1 = values[i]
                s2 = score - k1
                ret = twoSum(values, i+1, end, s2)
                for v in ret:
                    p1, p2 = v
                    res.append([k1, p1, p2])
            return res

        res = list()
        count = len(values)
        for i in range(count):
            if i > 0 and values[i] == values[i-1]:
                continue

            k1 = values[i]
            s3 = target - k1
            ret = threeSum(values, i+1, count-1, s3)
            for v in ret:
                p1, p2, p3 = v
                res.append([k1, p1, p2, p3])

        return res

    def wordBreak(self, s, word_dict):
        count = len(s)
        for i in range(count):
            x = s[:i+1]
            if x in word_dict:
                if i+1 >= count:
                    return True
                elif self.wordBreak(s[i+1:], word_dict) == True:
                    return True
        return False


    def wordBreak2(self, s, word_dict):
        for word in word_dict:
            count = len(word)
            if count > len(s):
                continue
            if s[:count] == word:
                if count == len(s):
                    return True
                elif self.wordBreak2(s[count:], word_dict) == True:
                    return True
        return False


    def wordBreak3(self, s, word_dict):
        dp = [False] * len(s)
        for word in word_dict:
            if s.startswith(word):
                dp[len(word)-1] = True

        for j in range(len(s)):
            for w in word_dict:
                p = j - len(w)
                k = s[p+1:j+1]
                if p >= 0 and k == w:
                    dp[j] = dp[j] or dp[p]
        return dp[-1]


    def romanToInt(self, s):
        count = len(s)
        ret = 0
        values = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        i = 0
        while i < count:
            if s[i] not in ('I', 'X', 'C') or i + 1 >= count:
                ret += (values[s[i]])
                i = i + 1
            else:
                if (s[i] == 'I' and s[i+1] in ('V', 'X')) \
                        or (s[i] == 'X' and s[i+1] in ('L', 'C')) \
                        or (s[i] == 'C' and s[i+1] in ('D', 'M')):
                    ret += (values[s[i+1]] - values[s[i]])
                    i = i + 2
                else:
                    ret += values[s[i]]
                    i = i + 1
        return ret

    def generate(self, numRows):
        x1 = [1]
        x2 = [1, 1]
        if numRows == 1:
            return [x1]

        if numRows == 2:
            return [x1, x2]

        ret = [x1, x2]
        for i in range(2, numRows):
            x = ret[-1]
            num = len(x)
            y = list()
            y.append(x[0])
            for j in range(num-1):
                a = x[j] + x[j+1]
                y.append(a)
            y.append(x[-1])
            ret.append(y)

        return ret

    def isMatch(self, s, p):
        n1 = len(s)
        n2 = len(p)

        if n1 == 0 and n2 == 0:
            return True

        if n1 > 0 and n2 == 0:
            return False

        if n1 == 0 and n2 > 0:
            if len(set(p)) == 1 and p[0] == '*':
                return True
            else:
                return False

        dp = list()
        for i in range(n1):
            value = list()
            for j in range(n2):
                value.append(False)
            dp.append(value)

        if p[0] == s[0] or p[0] == '?':
            dp[0][0] = True

        if p[0] == '*':
            for j in range(n1):
                dp[j][0] = True

        k = -1
        for i in range(n2):
            if p[i] != '*':
                k = i
                break
            dp[0][i] = True

        if k >= 0 and p[k] == '?':
            dp[0][k] = True

        for i in range(1, n1):
            for j in range(1, n2):
                if s[i] == p[j] or p[j] == '?':
                    dp[i][j] = dp[i][j] or dp[i-1][j-1]

                if p[j] == '*':
                    k = 0
                    while k < i and dp[k][j-1] == False:
                        k = k + 1
                        dp[i][j] = dp[i][j] or dp[k][j-1]
                    for t in range(k, j):
                        dp[t][j] = True

        return dp[n1-1][n2-1]


    def ReverseList(self, pHead):
        pReverseHead = None
        pNode = pHead
        prev = None
        while pNode:
            pNext = pNode.next
            if pNext is None:
                pReverseHead = pNode
            pNode.next = prev
            prev = pNode
            pNode = pNext
        return pReverseHead

    def reverseBetween(self, head, left, right):
        node = head
        k = 1
        front = None
        pLeft = None
        pRight = None
        back = None
        while node:
            if k == left - 1:
                front = node
            if k == left:
                pLeft = node
            if k == right:
                pRight = node
            if k == right + 1:
                back = node
            node = node.next
            k += 1

        pPrev = None
        pNode = pLeft
        while pNode:
            if pNode == back:
                break
            pNext = pNode.next
            pNode.next = pPrev
            pPrev = pNode
            pNode = pNext

        phead = head
        if front:
            front.next = pRight
        else:
            phead = pRight

        if back:
            pLeft.next = back
        return phead


    def LRU(self, operators, k):
        # write code here
        cache = dict()
        sequences = list()
        ret = list()
        num = len(operators)
        for i in range(num):
            pair = operators[i]
            if pair[0] == 2:
                key = str(pair[1])
                if key in cache.keys():
                    sequences.remove(key)
                    ret.append(cache[key])
                    sequences = sequences + [key]
                else:
                    ret.append(-1)
            else:
                key = str(pair[1])
                if len(sequences) == k:
                    cache.pop(sequences[0])
                    sequences = sequences[1:]

                sequences.append(key)
                value = pair[2]
                cache[key] = value
        return ret, sequences

    def reverse(self, x):
        # write code here
        values = str(abs(x))
        cnt = len(values)
        x2 = ''.join([values[cnt - 1 - i] for i in range(cnt)])
        x3 = float(x2)
        x3 = x3 if x > 0 else -x3
        if x3 < (-2 ** 31) or x3 > 2 ** 31 - 1:
            return -1
        return int(x3)


    def contain(self, s1, t1):
        def get_dict(s):
            ds = dict()
            for ch in s:
                if ch not in ds:
                    ds[ch] = 0
                ds[ch] += 1
            return ds

        ds = get_dict(s1)
        dt = get_dict(t1)
        dk = dict()

        tag = True
        for k in dt.keys():
            if k not in ds.keys():
                tag = False
            else:
                dk[k] = ds[k]
                if ds[k] < dt[k]:
                    tag = False
        return dk, dt, tag


    def minWindow(self, S, T):
        dS, dT, tag = self.contain(S, T)
        if not tag:
            return ''

        count = len(S)
        param = [count, 0, count]
        for ileft in range(0, count):
            c1 = S[ileft]
            if c1 not in dS:
                if 0 < count -1 - ileft < param[0]:
                    param = [count-1-ileft, ileft+1, count]
                continue

            records = list()
            for iright in range(count-1, ileft, -1):
                c2 = S[iright]
                if c2 not in dS:
                    if 0 < iright - ileft < param[0]:
                        param = [iright - ileft, ileft, iright]
                    continue

                dS[c2] -= 1
                records.append(c2)
                if dS[c2] >= dT[c2]:
                    if 0 < iright - ileft < param[0]:
                        param = [iright - ileft, ileft, iright]
                else:
                    break

            for ch in records:
                dS[ch] += 1

            dS[c1] -= 1
            if dS[c1] >= dT[c1]:
                if 0 < count - 1 - ileft < param[0]:
                    param = [count - 1 - ileft, ileft + 1, count]
            else:
                break

        _, left, right = param
        return S[left:right]


    def qsort_recursive(self, nums, left, right):
        if left >= right:
            return

        i = left
        j = right
        while i < j:
            while nums[j] >= nums[i] and j > i:
                j = j - 1

            if j > i:
                tmp = nums[j]
                nums[j] = nums[i]
                nums[i] = tmp

            while nums[i] <= nums[j] and i < j:
                i = i + 1

            if i < j:
                tmp = nums[i]
                nums[i] = nums[j]
                nums[j] = tmp

        self.qsort_recursive(nums, left, i-1)
        self.qsort_recursive(nums, i+1, right)


    def qsort(self, nums):
        self.qsort_recursive(nums, 0, len(nums) - 1)
        return nums

    def numTrees(self, n):
        if n == 1:
            return 1
        if n == 2:
            return 2

        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2

        for i in range(3, n+1):
            for k in range(i):
                left = k
                right = i - 1 - k
                dp[i] += dp[left] * dp[right]

        return dp[n]


    def add(self, root, k):
        if root:
            root.val += k
            self.add(root.left, k)
            self.add(root.right, k)


    def generateTrees(self, n):
        n1 = TreeNode(1)
        if n == 1:
            return [n1]

        n2 = TreeNode(2, left=TreeNode(1))
        n3 = TreeNode(1, right=TreeNode(2))
        if n == 2:
            return [n2, n3]

        dp = list()
        dp.append([])
        dp.append([n1])
        dp.append([n2, n3])

        for i in range(3, n+1):
            di = list()
            for k in range(i):
                lefts = copy.deepcopy(dp[k])
                rights = copy.deepcopy(dp[i-1-k])
                for index, node in enumerate(rights):
                    self.add(node, k+1)

                if len(lefts) == 0:
                    for r in rights:
                        nk = TreeNode(k+1)
                        nk.right = copy.deepcopy(r)
                        di.append(nk)

                if len(rights) == 0:
                    for l in lefts:
                        nk = TreeNode(k+1)
                        nk.left = copy.deepcopy(l)
                        di.append(nk)

                for l in lefts:
                    for r in rights:
                        nk = TreeNode(k+1)
                        nk.left = copy.deepcopy(l)
                        nk.right = copy.deepcopy(r)
                        di.append(nk)

            dp.append(di)
        return dp[-1]


    def maxProfit2(self, prices):
        num = len(prices)
        if num <= 1:
            return 0

        dp = [0] * num
        dp[1] = max(prices[1] - prices[0], 0)

        for i in range(2, num):
            pi = prices[i]
            for j in range(i):
                pj = prices[j]
                value = pi - pj
                last = 0
                if j > 0:
                    last = max(dp[:j])
                if last + value > dp[i]:
                    dp[i] = last + value
        return max(dp)


    def isSubsequence(self, s: str, t: str):
        if s == '':
            return True

        nt = len(t)
        k = 0
        for i in range(nt):
            if t[i] == s[k]:
                k = k + 1
            if k == len(s):
                return True

        return False

    def sqrt_taylor(self, x, seed):
        def n_deravative(n, x0):
            heads = 0.5 - n
            s0 = 0.5
            for i in range(1, n):
                s0 *= (1 - 2 * i) / 2.0
            s1 = s0 * math.pow(x0, heads)
            return s1

        N = 30
        f = math.sqrt(seed)
        level = 1.0
        for i in range(1, N):
            level = level * i
            der = n_deravative(i, seed)
            dx = (x - seed) ** i
            f += der * dx / level
        return f


    def divide(self, dividend, divisor):
        sign = 1
        if divisor * dividend < 0:
            sign = -1

        f1 = dividend if dividend > 0 else -dividend
        f2 = divisor if divisor > 0 else -divisor

        if f2 == 1:
            return f1 * sign

        if f1 == f2:
            return sign

        if f1 < f2:
            return 0

        v1 = f1 // 2 + 1
        high = v1
        low = 1
        while low < high:
            mid = (low + high) // 2
            if mid * f2 == f1 or (mid * f2 < f1 and (mid + 1) * f2 > f1):
                return mid * sign

            if (mid + 1) * f2 == f1:
                return (mid + 1) *sign

            if (mid - 1) * f2 == f1 or (mid * f2 > f1 and (mid - 1) * f2 < f1):
                return (mid - 1) * sign

            if mid * f2 < f1:
               low = mid

            if mid * f2 > f1:
                high = mid

        return 1


    def combinationSum2(self, candidates, target):
        values = sorted(candidates)
        count = len(values)

        solutions = set()
        def dfs(pos, score, path):
            if score == 0:
                solutions.add(tuple(path[:]))
                return True

            if len(path) > 0 and path[-1] > score:
                return False

            visited = set()
            for i in range(pos, count):
                if values[i] in visited:
                    continue
                visited.add(values[i])
                path.append(values[i])
                index = len(path) - 1
                dfs(i+1, target-sum(path), path)
                path.pop(index)

        dfs(0, target, [])
        return list(solutions)


    def removeElements(self, head, val):
        node = head
        prev = None
        ret = head
        while node:
            pnext = node.next
            if node.val == val:
                if prev:
                    prev.next = pnext
                else:
                    ret = pnext
            else:
                prev = node
            node = pnext
        return ret


    def isValidBST(self, root):
        if root is None:
            return True

        left = root.left
        right = root.right

        if left:
            pleft = left
            while pleft:
                max_left = pleft.val
                pleft = pleft.right

            if max_left >= root.val:
                return False

        if right:
            pright = right
            while pright:
                min_right = pright.val
                pright = pright.left
            if min_right <= root.val:
                return False

        bleft = self.isValidBST(left)
        bright = self.isValidBST(right)
        return bleft and bright


    def recoverTree(self, root):
        def visited(root, values, nodes):
            if root is None:
                return

            visited(root.left, values, nodes)
            values.append(root.val)
            nodes.append(root)
            visited(root.right, values, nodes)
        vals = []
        nodes = []
        visited(root, vals, nodes)
        vals.sort()
        for index, p in enumerate(nodes):
            p.val = vals[index]


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

    print(' -- delete duplicates --')
    [1, 1, 2, 3, 3]
    k1 = ListNode(1)
    k1.next = ListNode(1)
    k1.next.next = ListNode(2)
    k1.next.next.next = ListNode(3)
    k1.next.next.next.next = ListNode(3)
    value = solution.deleteDuplicates(k1)


    print(' -- findSubstring -- ')
    # s = "barfoothefoobarman"
    s = "barfoofoobarthefoobarman"
    # words = ["foo","bar"]
    words = ["bar","foo","the"]
    value = solution.findSubstring(s, words)
    print(value)

    print(' -- levelOrderBottom -- ')
    k1 = TreeNode(3)
    k1.left = TreeNode(9)
    k1.right = TreeNode(20)
    k1.right.left = TreeNode(15)
    k1.right.right = TreeNode(7)

    value = solution.levelOrderBottom(k1)
    print(value)

    print(' -- maxProfit -- ')
    values = [7,1,5,3,6,4]
    value = solution.maxProfit(values)
    print(value)


    print(' -- reverse list --')
    k1 = ListNode(5)
    k1.next = ListNode(6)
    k1.next.next = ListNode(4)
    k1.next.next.next = ListNode(9)
    kn = solution.reverse_list(k1)
    while kn:
        print(kn.val)
        kn = kn.next


    print('-- groupAnagrams ')
    values = ["eat","tea","tan","ate","nat","bat"]
    # values = ['']
    ret = solution.groupAnagrams2(values)
    print(ret)


    print('-- four sums ')
    # values = [1, 0, -1, 0, -2, 2]
    values = [1,-2, -5, -4, -3, 3, 3, 5]
    target = -11
    ret = solution.fourSum(values, target)
    print(ret)

    print('-- word break ')
    s = "applepenapple"
    wordDict = ["apple","pen"]
    ret = solution.wordBreak3(s, wordDict)
    print(ret)


    print('-- romanToInt ')
    s = "MCMXCIV"
    ret = solution.romanToInt(s)
    print(ret)


    print('-- generate ')
    ret = solution.generate(5)
    print(ret)

    print('-- generate ')
    # ret = solution.isMatch('abcabc', '*abc')
    ret = solution.isMatch( "abcabczzzde", "*abc???de*")
    print(ret)

    print('-- LRU ')
    operators = [[1,1,1],[1,2,2],[1,3,2],[2,1],[1,4,4],[2,2]]
    ret, seqs = solution.LRU(operators, 3)
    print(ret)
    print(seqs)


    print('-- Reverse ')
    w = solution.reverse(2123456789)
    print(w)


    print('-- minWindow --')
    # w = solution.minWindow("XDOYEZODEYXNZ","XYZ")
    w = solution.minWindow("cabwefgewcwaefgcf","cae")
    print(w)

    print('-- qsort --')
    # w = solution.minWindow("XDOYEZODEYXNZ","XYZ")
    w = solution.qsort([5,7,2,4,7,8,1,9,12,19,2,1,3,8])
    print(w)


    print('-- numTrees --')
    num = solution.numTrees(9)
    print(num)

    print('-- generateTrees --')
    ret = solution.generateTrees(5)
    print(ret)

    print('-- maxProfit -- ')
    prices = [7,1,5,3,6,4]
    ret = solution.maxProfit2(prices)
    print(ret)

    print('-- sqrt taylor -- ')
    ret = solution.sqrt_taylor(2, 1.44)
    print(ret)


    print('-- divide --')
    ret = solution.divide(13, 3)
    print(ret)

    print('-- reverseBetween --')
    k1 = ListNode(5, next=ListNode(2, next=ListNode(3, next=ListNode(9, next=ListNode(8)))))
    k1= solution.reverseBetween(k1, 1,4)
    while k1:
        print(k1.val)
        k1 = k1.next


    print("-- nextPermutation --")
    nums = [5,4,7,5,3,2]
    solution.nextPermutation(nums)
    print(nums)


    print('-- combinationSum2 --')
    candidates = [14,6,25,9,30,20,33,34,28,30,16,12,31,9,9,12,34,16,25,32,8,7,30,12,33,20,21,
                  29,24,17,27,34,11,17,30,6,32,21,27,17,16,8,24,12,12,28,11,33,10,32,22,13,34,18,12]
    target = 27
    ret = solution.combinationSum2(candidates, target)
    print(ret)

    print('-- removeElements --')
    k1 = ListNode(5, next=ListNode(5, next=ListNode(5, next=ListNode(9, next=ListNode(8)))))
    ret = solution.removeElements(k1, 5)
    print(ret)


    print('-- recover tree --')
    k1 = TreeNode(3, left=TreeNode(1), right=TreeNode(5, left=TreeNode(4)))
    ret = solution.recoverTree(k1)
    print(ret)


def debug():
    k1 = ListNode(5)
    k2 = ListNode(5)
    k3 = k1
    sample = [k1, k2]

    print(k3 in sample)

    n1 = TreeNode(2, left=TreeNode(1, left=TreeNode(5)), right=TreeNode(2))
    n2 = copy.deepcopy(n1)
    n1.left.left.val = 10

    print(n2.left.left.val)
    print(n1.left.left.val)



if __name__ == '__main__':
    main()
    # debug()


        
