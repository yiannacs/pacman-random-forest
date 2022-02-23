# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np

class Classifier:
    def __init__(self):
        self.root = TreeNode()

    def reset(self):
        pass
    
    def fit(self, data, target):
        # data = self.parseData(data)
        data = np.array(data)
        idx = np.array(range(data.shape[1])).reshape(1, data.shape[1])
        data = np.concatenate((idx, data), axis=0)
        target = np.array(target)
        
        self.root = self.buildTree(data, target)
        
        print()
        print('*************************************************')
        self.printTree(self.root)
        # print(data.shape)
        
    def buildTree(self, data, target):
        # For testing
        # target = np.array([1,1,1,1,1])
        print('Building node')
        print('Remaining attributes: {}'.format(data.shape[1]))
        print('Remaining data:')
        print(data)
        print('Remaining target:')
        print(target)
        print('')
        
        # if data.shape[0] == 7:
        #     print('I\'m in')
        #     print(data)
        #     print(target)
        
        print('Holi1')
        # Check if target has all equal values:
        if np.equal(target, target[0]).all():
            return TreeNode(value=target[0])
            # print(node.value)
            
        print('Holi2')
        # No attributes left
        if data.shape[1] == 0:
            # Use most common target value
            return TreeNode(value=np.argmax(np.bincount(target)))
        
        print('Holi3')
        # Attributes left, but remaining data is all the same
        if np.equal(data[1:,:], data[1]).all():
            print('DATA IS ALL THE SAME')
            print(data)
            print(target)
            # Use most common target value
            return TreeNode(value=np.argmax(np.bincount(target)))

        
        print('Holi4')
        
        # Find the best attribute left
        attr_max_gain_current_index = information_gain(data[1:, :], target)
        
        # Divide data by values of attr_max_gain
        attr_zeros, target_zeros, attr_ones, target_ones = self.branchOutAttribute(attr_max_gain_current_index,
                                                                               data,
                                                                               target)
        
        print(attr_max_gain_current_index)
        print(data[0][attr_max_gain_current_index])
        
        # If case attribute==0 is empty, set to most frequent value
        if len(attr_zeros) == 0:
            left = TreeNode(value=np.argmax(np.bincount(target_zeros)))
        else:
            # Still have data, recursively built this branch
            left = self.buildTree(attr_zeros, target_zeros)
        
        print('Gonna do right now')
        # If case attribute==1 is empty, set to most frequent value
        if len(attr_ones) == 0:
            right = TreeNode(value=np.argmax(np.bincount(target_ones)))
        else:
            # Still have data, recursively built this branch
            right = self.buildTree(attr_ones, target_ones)
            
        return TreeNode(value=data[0][attr_max_gain_current_index], attr_false=left, attr_true=right)
        
            
        # print('branches of first node')
        # print(attr_zeros.shape)
        # print(target_zeros)
        # print(attr_ones.shape)
        # print(target_ones)
        # np.mask_rows(attr_zero)
        # np.mask_rows(attr_ones)
    
    def printTree(self, root):
        if root is None:
            return
        
        queue = []
        
        queue.append(root)
        
        while(len(queue) > 0):
            print(queue[0].value)
            node = queue.pop(0)
            
            if node.attr_false is not None:
                queue.append(node.attr_false)
                
            if node.attr_true is not None:
                queue.append(node.attr_true)
                
        
        
    def branchOutAttribute(self, attr, data, target):
        # Get mask to keep rows where attribute value is zero
        mask_zeros = np.ma.masked_where(data[1:,attr]==1, data[1:, attr]).mask
        if isinstance(mask_zeros,  np.bool_):
            mask_zeros = np.zeros(len(data[1:]))
        mask_ones = np.logical_not(mask_zeros)
        
        # Mask target before reshaping mask
        target_zeros = np.ma.MaskedArray(target, mask=mask_zeros)
        target_ones = np.ma.MaskedArray(target, mask=mask_ones)
        # print(attr)
        
        mask_zeros = np.concatenate(([0], mask_zeros))
        mask_ones = np.concatenate(([0], mask_ones))
        
        # if isinstance(mask_zeros,  np.bool_):
        #     print('Data:')
        #     print(data)
        #     print()
        #     print('Target:')
        #     print(target)
        #     print()
        #     print(mask_zeros)
        
        # Get right shape
        mask_zeros = np.concatenate((mask_zeros.reshape(len(mask_zeros),1), np.zeros((len(mask_zeros), data.shape[1] - 1))), axis=1)
        mask_ones = np.concatenate((mask_ones.reshape(len(mask_ones),1), np.zeros((len(mask_ones), data.shape[1] - 1))), axis=1)
    
        # print(mask_zeros)
        # print(mask_ones)
        
        # Create masked arrays with full data and correctly-shaped masks
        attr_zeros = np.ma.MaskedArray(data, mask = mask_zeros)
        attr_ones = np.ma.MaskedArray(data, mask = mask_ones)
        
        # Separate attribute = 0 and = 1
        attr_zeros = np.ma.compress_rowcols(attr_zeros, axis=0)
        target_zeros = np.ma.compressed(target_zeros)
        
        attr_ones = np.ma.compress_rowcols(attr_ones, axis=0)
        target_ones = np.ma.compressed(target_ones)
        
        # Remove attribute from data
        attr_zeros = np.delete(attr_zeros, attr, axis=1)
        attr_ones = np.delete(attr_ones, attr, axis=1)
        
        return attr_zeros, target_zeros, attr_ones, target_ones        
            
    def predict(self, data, legal=None):
        return 1
    
    def parseData(self, data):
        # print('ruifhceruifhceuij')
        data = np.array(data)
        for i in range(len(data)):
            # val = 1
            walls = np.zeros((len(data)))
            for j in range(0, 4):
                walls = (walls*2) + data[:,j]
                
            food = np.zeros((len(data)))
            for j in range(4, 8):
                food = (food*2) + data[:,j]
                
            ghost1 = np.zeros((len(data)))
            for j in range(8, 16):
                ghost1 = (ghost1*2) + data[:,j]
                
            ghost2 = np.zeros((len(data)))
            for j in range(16, 24):
                ghost2 = (ghost2*2) + data[:,j]
                
            ghostVisible = np.zeros((len(data)))
            ghostVisible = ghostVisible + data[:,24]        
        
        return np.array([walls, food, ghost1, ghost2, ghostVisible]).T
    
class TreeNode():
    def __init__(self, value=0, attr_false=None, attr_true=None):
        self.value = value
        self.attr_false = attr_false
        self.attr_true = attr_true
    
def entropy(target):
    # Count occurrences of each outcome
    values, count = np.unique(target, return_counts=True)
    # print('Entropy')
    # print(values)
    # print(count)
    
    # Compute probability for each outcome
    p_i = count/len(target)
    
    # print(p_i)
    
    # 0 yields nan when computing log2
    # Replace with 1. Since log2(1) = 0,
    # 0*log2(0) = 0 is kept
    p_i = np.ma.masked_equal(p_i, 0).filled(1)
    
    entropy = np.sum(-p_i*np.log2(p_i))
        
    # print()
    return entropy
        
def information_gain(data, target, data_values=None, target_values=None):
    entropy_s = entropy(target)
    
    entropy_a = np.zeros(data.shape[1])
    # For each feature:
    for i in range(data.shape[1]):
        # print()
        # print('i: {}'.format(i))
        # Count value occurrences
        feature_values, feature_count = np.unique(data[:, i], return_counts=True)
        
        sum_entropy_v = 0
        # For each value of the feature
        for j in range(len(feature_count)):
            # print('{}\t{}'.format(feature_values[j], feature_count[j]))
            target_v = np.ma.masked_where(data[:, i] != feature_values[j], target)
            # print(target_v)
            target_v = target_v.compressed()
            entropy_v = entropy(target_v)
            
            p_v = feature_count[j]/data.shape[0]
            
            sum_entropy_v += p_v * entropy_v
            
        entropy_a[i] = entropy_s - sum_entropy_v
    # print('Information gain')
    # print(np.argmax(entropy_a))
    # print(entropy_s)
    # print(entropy_a)
    # print()
    
    return np.argmax(entropy_a)

def runTests():
    # Testint entropy
    t1 = np.array([2,3,0,2,1])  # 1.921928
    t2 = np.array([3,0,0,2,1])  # 1.921928
    t3 = np.array([0,2,2,0,2])  # 0.970951
    t4 = np.array([3,3,3,2,0])  # 1.370951
    t5 = np.array([2,1,3,3,2])  # 1.521928
    
    print(entropy(t1))
    print(entropy(t2))
    print(entropy(t3))
    print(entropy(t4))
    print(entropy(t5))
    print()
            
    # Testing information gain
    data = np.array([
        [0,0,0],
        [0,1,1],
        [1,0,2],
        [1,1,1],
        [0,1,0],
        [0,1,2],
        [0,2,1],
        [1,1,0],
        [1,0,2],
        [1,2,0]
        ])
    target = np.array([1,1,0,1,1,1,0,0,1,0])
    information_gain(data, target)
    print()
    
    data = np.array([
        [1,1,1,1],
        [1,1,1,2],
        [2,1,1,1],
        [3,2,1,1],
        [3,3,2,1],
        [3,3,2,2],
        [2,3,2,2],
        [1,2,1,1],
        [1,3,2,1],
        [3,2,2,1],
        [1,2,2,2],
        [2,2,1,2],
        [2,1,2,1],
        [3,2,1,2]
        ])
    target = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
    information_gain(data, target)
    print()
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    











