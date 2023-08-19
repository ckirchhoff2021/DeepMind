class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

     @staticmethod
     def create(values):
         N = len(values)
         root = TreeNode(values[0])
         candidates = [root]
         i = 0
         while i < N:
            Nums = len(candidates)
            nodes = list()
            for j in range(Nums):
                i = i + 1
                if i < N and values[i]:
                    node = TreeNode(values[i])
                    nodes.append(node)
                    candidates[j].left = node
                i = i + 1
                if i < N and values[i]:
                    node = TreeNode(values[i])
                    nodes.append(node)
                    candidates[j].right = node
            candidates = nodes
         return root

     @staticmethod
     def fr_visit(root):
         if root:
            print(root.val, end=' ')
            TreeNode.fr_visit(root.left)
            TreeNode.fr_visit(root.right)

if __name__ == '__main__':
  values = [5,4,8,11,None,13,4,7,2,None,None,5,1]
  root = TreeNode.create(values)
  print(root)
  TreeNode.fr_visit(root)
