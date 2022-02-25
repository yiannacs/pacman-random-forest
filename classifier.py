# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
from sklearn.model_selection import train_test_split
import copy
import random

class Classifier:
    def __init__(self):
        self.root = TreeNode()
        self.default = None

    # TODO Do I need something here?
    def reset(self):
        pass
    
    def fit(self, data, target):
        """
        Fit tree to data by doing the following steps. 
            
        1. Build tree
        2. Rule post-prunning 
        3. Sort rules by certainty factor
        4. Select a default class
                
        Mainly based on ID3 and C4.5 but, for simplicity,
        also other work by Quinlan.
            
        Parameters
        ----------
        data : numpy.darray
            Training set data
        target : numpy.darray
            Classes for training data

        Returns
        -------
        None.

        """
        
        # Convert data to numpy array
        data = np.array(data)
        target = np.array(target)
        
        # Not splitting data
        data_train = data
        target_train = target
        
        # Split data
        # data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.15)
        
        # Add headers to training data:
        # [0, 1, 2, ..., number_of_attributes - 1]
        # This makes it easier to keep indexes when splitting/taking attributes out
        idx = np.array(range(data.shape[1])).reshape(1, data.shape[1])
        data_train = np.concatenate((idx, data_train), axis=0)
        
        # Build tree
        self.root = self.buildTree(data_train, target_train)
        
        # Rule post-prunning
        self.rulePostPrunning(data_train[1:len(data_train), :], target_train)
        
        # Sort by certainty factor (Quinlan, 1987a)
        self.paths.sort(reverse=True, key=lambda x: x['cf'])
        
        # Find default path  
        self.default = self.findDefault(data_train[1:len(data_train), :], target)
        
    def predict(self, data, legal=None):
        """
        Predict good move given state defined by data

        Parameters
        ----------
        data : list
            List defining current state of the game.
        legal : list, optional
            List of legal actions. The default is None.

        Returns
        -------
        move : int
            Prediction of a good move given state in data.

        """
        
        data = np.array(data)
        
        # Predict using simplified rules table
        move = self.predictWithRules(data)
        
        return move    
    
    def buildTree(self, data, target):
        """
        Recursively builds decission tree based on data, target
        following the ID3 algorithm (Quinlan, 1986).

        Parameters
        ----------
        data : numpy.ndarray
            Training set data.
        target : numpy.ndarray
            Classes for training data.

        Returns
        -------
        NodeTree
            Root of tree.

        """
       
        # For given data, check if all cases have the same class
        if np.equal(target, target[0]).all():
            # Return said class
            return TreeNode(value=target[0])
        
        # FIXME This is wrong. Should be handled right after partitioning data
        # No cases left for this attribute
        if data.shape[1] == 0:
            # Use most common target value
            print('DOES THIS ACTUALLY HAPPEN?????')
            return TreeNode(value=np.argmax(np.bincount(target)))
        
        # All remaining test cases are equal
        # Excluding first row from check as those are headers
        if np.equal(data[1:,:], data[1]).all():
            # Return most common class
            return TreeNode(value=np.argmax(np.bincount(target)))

        # Find the best attribute left
        # All features are binary, so information gain is a good metric
        # to determine importance of features.
        # Need only the index in the current array, so don't need to pass headers
        attr_max_gain_current_index = information_gain(data[1:, :], target)
        
        # Divide data by values of attr_max_gain
        attr_zeros, target_zeros, attr_ones, target_ones = self.branchOutAttribute(attr_max_gain_current_index,
                                                                               data,
                                                                               target)
        
        # No cases when attribute == 0
        if len(target_zeros) == 0:
            # Set branch to leaf with class given by the most common class at this node
            left = TreeNode(value=np.argmax(np.bincount(target_zeros)))
        # Still have data,
        else:
            # recursively built this branch
            left = self.buildTree(attr_zeros, target_zeros)

        # Determine if left branch points to leaf            
        if left.attr_false is None and left.attr_true is None:
            left_is_leaf = True
        else:
            left_is_leaf = False

        # No cases with attribute == 0
        if len(target_ones) == 0:
            # Set branch to leaf with class given by the most common class at this node
            right = TreeNode(value=np.argmax(np.bincount(target)))
        # There are cases where attribute is 1
        else:
            # Recursively build right branch
            right = self.buildTree(attr_ones, target_ones)

        # Determine if right branch points to a leaf 
        if right.attr_false is None and right.attr_true is None:
            right_is_leaf = True
        else:
            right_is_leaf = False
        
        # If both branches point to leaves,
        # and both leaves have the same class,
        if right_is_leaf and left_is_leaf:
            if right.value == left.value:
                # Discard leaves and set this node to said class
                return TreeNode(value=right.value)
            
        # Note a leaf, return node with branch pointing to subtrees
        return TreeNode(value=data[0][attr_max_gain_current_index], attr_false=left, attr_true=right)

    def rulePostPrunning(self, data, target):
        """
        Converts original tree to list of rules, tests their significancy and
        removes them based on the certainty factor used in (Quinlan, 1987b)
        
        Parameters
        ----------
        data : numpy.ndarray
            Training set data.
        target : numpy.ndarray
            Classes for training set.

        Returns
        -------
        None.

        """
        
        paths = []
        
        # Generate rule table
        path = []
        self.traverse(self.root, paths, list(path))
        
        # Remove rules
        for i in range(len(paths)):
            for step in paths[i]['path']:
                # Deep copy to keep original intact when removing preconditions
                alt_path = copy.deepcopy(paths)
                
                # Remove current precondition from rule
                alt_path[i]['path'].remove(step)
                
                # Build contingency table
                xy = np.concatenate((data, target.reshape(1, len(target)).T), axis=1)
                
                # Separate cases that belong and not belong to the rule's class
                belong_to_class = np.array([xy[l, :] for l in range(xy.shape[0]) if xy[l, -1]==paths[i]['target']])
                not_belong_to_class = np.array([xy[l, :] for l in range(xy.shape[0]) if xy[l, -1] != paths[i]['target']])
                
                # Count combinations of [not] belonging to the class and [not] satisfying the rule
                belong_satisfy = np.sum(belong_to_class[:, step[0]])
                belong_not_satisfy = belong_to_class.shape[0] - belong_satisfy
                not_belong_satisfy = np.sum(not_belong_to_class[:, step[0]])
                not_belong_not_satisfy = not_belong_to_class.shape[0] - not_belong_satisfy
                
                # Compute certainty factor for original and alternate rule
                certainty = self.computeCertainty(belong_satisfy, not_belong_satisfy)
                certainty_alt = self.computeCertainty(belong_satisfy + belong_not_satisfy, not_belong_satisfy + not_belong_not_satisfy)
                
                # Set the rule's certainty factor to the greater one
                paths[i]['cf'] = max([certainty, certainty_alt])
                
                # Keep the rule with the greater certainty factor
                if certainty_alt >= certainty:
                    paths[i]['path'].remove(step)
        
        # Set the tree's rule table
        self.paths = paths

    def findDefault(self, data, target):
        """
        Find class to be returned if a certain datum does not fit any path
        Per (Quinlan 1992), the default class is the most frequent class
        in the subset of training cases that don't fit any path.

        Parameters
        ----------
        data : numpy.ndarray
            Training set data
        target : numpy.ndarray
            Classes for training set data

        Returns
        -------
        int
            Class to set as default

        """
                
        predictions = self.predictWithRulesBatch(data)
        
        # Get classes from training cases that yield no answer
        unpredicted_targets = target[np.argwhere(predictions==None)]
        
        # If all cases had a rule, return most common class
        if len(unpredicted_targets) < 1:
            return np.argmax(np.bincount(target))
        
        # Return most common class among unpredicted cases
        return np.argmax(np.bincount(unpredicted_targets))
        
    def discardRuleSupersets(self, paths, target):
        """
        For each possible class, remove rules that are supersets of simpler rules.
        
        Currently not used

        Parameters
        ----------
        target : numpy.ndarray
            Target from which to remove supersets

        Returns
        -------
        None.

        """
        
        found_target_i = False
        for path_i in self.paths:
            if path_i['target'] == target:
                found_target_i = True
                
                # Eat up overhead of looping from the start to not deal with
                # updating indices
                found_target_j = False
                for path_j in self.paths:
                    if path_j['target'] == target:
                        found_target_j = True
                
                        # Check if any of pair is subset of another,
                        # delete superset
                        if path_i != path_j:
                            if set(path_i['path']).issubset(path_j['path']):
                                self.paths.remove(path_j)
                                
                            if set(path_j['path']).issubset(path_i['path']):
                                self.paths.remove(path_i)
                                
                    elif found_target_j:
                        break
            elif found_target_i:
                break
                    
    def computeCertainty(self, y1, e1):
        """
        Compute certainty given y1, e1 as used in (Quinlan, 1987b)

        Parameters
        ----------
        y1 : int
            number of correct cases.
        e1 : int
            total number of cases.

        Returns
        -------
        float
            certainty factor.

        """
        
        return (y1 - 1/2) / (y1 + e1)
        
    def predictWithRulesBatch(self, data, alt=None):
        """
        Generates predictions from rules table for many rows of input data.
        
        If provided, uses alt instead of intance table.

        Parameters
        ----------
        data : numpy.ndarray
            Array with rows cases to predict.
        alt : list(dict({'path'})), optional
            Alternate rules table. The default is None.

        Returns
        -------
        numpy.ndarray
            Predicted classes for rows of data.

        """
        
        predictions = []
        for row in data:
            pred_i = self.predictWithRules(row, alt)
            predictions.append(pred_i)
            
        return np.array(predictions)
        
    def predictWithRules(self, data, alt=None):
        """
        Generate prediction from rules table for a single datum

        If provided, uses alt instead of intance table.

        Parameters
        ----------
        data : numpy.ndarray
            State for which to predict.
        alt : list(dict({'path'})), optional
            Alternate rules table. The default is None.

        Returns
        -------
        numpy.ndarray
            Predicted class for data.

        """
       
        if alt is not None:
            for path in alt:
                # Go along each condition in current rule
                for step in (path['path']):
                    # If a condition is not met,
                    if data[step[0]] != step[1]:
                        # try the next rule
                        break
                    # If at the last step,
                    if step == path['path'][-1]:
                        # condition is met because loop didn't break,
                        # Return target for this rule
                        return path['target']
        else:
            for path in self.paths:
                # Go along each condition in current rule
                for step in (path['path']):
                    # If current condition is not met,
                    if data[step[0]] != step[1]:
                        # Break and go to next rule
                        break
                    # If at last step,
                    if step == path['path'][-1]:
                        # Loop didn't break, so condition is met
                        # Return target for this rule
                        return path['target']
        
        # No matching rule found
        return self.default
                        
    def traverse(self, node, paths, path):
        """
        Recursively travels along each possible path down to every leaf in the tree.
        Along the way, generate rules table.

        Parameters
        ----------
        node : TreeNode
            Starting node.
        paths : list
            List to which each complete path is appended.
        path : list
            Nodes (rules) in current path.

        Returns
        -------
        None.

        """
        
        # Node is not a leaf
        if node.attr_true is not None:
            # Take left path
            path_left = list(path)
            path_left.append((node.value, 0))
            self.traverse(node.attr_false, paths, path_left)

            # Take right path
            path_right = list(path)
            path_right.append((node.value, 1))                
            self.traverse(node.attr_true, paths, path_right)
            
        # Node is a leaf
        if node.attr_true is None:
            paths.append({'path': path, 'target': node.value})
        
    def printTree(self, node, depth=0):
        """
        Prints tree by printing each node's value, whether it's an attribute
        index or class, and depth of the node.

        Parameters
        ----------
        node : TreeNode
            Starting node from which to traverse the tree.
        depth : int, optional
            Depth of the current node. The default is 0.

        Returns
        -------
        None.

        """
        
        # Node is not a leaf
        if (node.attr_false is not None) or (node.attr_true is not None):
            # Print current node as an attribute index
            print('{}\tindex = {}'.format(depth, node.value))
            
            # Keep traversing tree
            self.printTree(node.attr_false, depth+1)
            self.printTree(node.attr_true, depth+1)
        else:
            # Print node as a class
            print('{}\ttarget = {}'.format(depth, node.value))
        
    def branchOutAttribute(self, attr, data, target):
        """
        Splits data and target into two sets, one for cases where the value
        of the attribute at index attr is 0, and one for cases where it's 1.

        Parameters
        ----------
        attr : int
            Index of the attribute with respect to which split cases.
        data : numpy.ndarray
            Cases to divide.
        target : TYPE
            Classes corresponding to each case in data.

        Returns
        -------
        attr_zeros : numpy.ndarray
            Cases data where attribute is 0.
        target_zeros : numpy.ndarray
            Classes corresponding to cases where attribute is 0.
        attr_ones : numpy.ndarray
            Cases data where attribute is 1.
        target_ones : numpy.ndarray
            Classes corresponding to cases where attribute is 1.

        """
        
        # Get mask that keeps rows where attribute value is 0
        # (exclude headers row)
        mask_zeros = np.ma.masked_where(data[1:,attr]==1, data[1:, attr]).mask
        # If the mask is empty, allow all values
        if isinstance(mask_zeros,  np.bool_):
            mask_zeros = np.zeros(len(data[1:]))
        
        # Get mask for cases when attribute is 1
        mask_ones = np.logical_not(mask_zeros)
        
        # Mask target before reshaping mask 
        target_zeros = np.ma.MaskedArray(target, mask=mask_zeros)
        target_ones = np.ma.MaskedArray(target, mask=mask_ones)
        
        # Modify mask to fit with data headers
        mask_zeros = np.concatenate(([0], mask_zeros))
        mask_ones = np.concatenate(([0], mask_ones))
        
        # Reshape to size of the full data array
        mask_zeros = np.concatenate((mask_zeros.reshape(len(mask_zeros),1), np.zeros((len(mask_zeros), data.shape[1] - 1))), axis=1)
        mask_ones = np.concatenate((mask_ones.reshape(len(mask_ones),1), np.zeros((len(mask_ones), data.shape[1] - 1))), axis=1)
        
        # Create masked arrays with full data and correctly-shaped masks
        attr_zeros = np.ma.MaskedArray(data, mask = mask_zeros)
        attr_ones = np.ma.MaskedArray(data, mask = mask_ones)
        
        # Keep only cases when attribute is 0
        attr_zeros = np.ma.compress_rowcols(attr_zeros, axis=0)
        target_zeros = np.ma.compressed(target_zeros)
        
        # Keep only cases when attribute is 1
        attr_ones = np.ma.compress_rowcols(attr_ones, axis=0)
        target_ones = np.ma.compressed(target_ones)
        
        # Remove attribute from data
        attr_zeros = np.delete(attr_zeros, attr, axis=1)
        attr_ones = np.delete(attr_ones, attr, axis=1)
        
        return attr_zeros, target_zeros, attr_ones, target_ones    

    def predictBatch(self, data, legal=None):
        """
        Predict for many rows of data by calling predict(self, data, legal)
        on each of them

        Parameters
        ----------
        data : numpy.ndarray
            Array with data cases.
        legal : list, optional
            List of legal actions. The default is None.

        Returns
        -------
        numpy.ndarray
            Predicted classes for each row of data.

        """
        
        moves = []
        
        # Call predict on each row
        for data_i in data:
            moves.append(self.predict(data_i))
            
        return np.array(moves)
                
    def traverseWithData(self, node, data):
        """
        Predict a good move based on the state of the game defined by data and
        by following the conditions in the (unprunned) decission tree.

        Parameters
        ----------
        node : TreeNode
            Starting node.
        data : numpy.ndarray
            Array defining state of the game.

        Returns
        -------
        int
            Good move predicted based on data.

        """
        
        # If node is a leaf, return value
        if node.attr_false is None and node.attr_true is None:
            return node.value
        
        # Node is not a leaf
        attribute = node.value
        # If value in data of node attribute is 1,
        if data[attribute]:
            # Traverse right branch
            return self.traverseWithData(node.attr_true, data)
        # If value in data of node attribute is 0,
        else:
            # Traverse left branch
            return self.traverseWithData(node.attr_false, data)
        
class TreeNode():
    """
    Class for tree nodes.
    
    Holds the index of the attribute to check when traversing the tree and
    passing through the node. Alternatively, if the node is a leaf, holds the
    target value.
    
    To determine whether a node is a leaf, simply check if it's branch
    attributes (or any of them, as in this case no node should have only
    one branch) are None.
    
    Attributes
    ----------
    value : int
        If the node is not a leaf, attribute to check when passing the node.
        If the node is a leaf, predicted target value.
    attr_false : TreeNode
        Root of subtree in left (i.e. value of node attribute is 0) branch.
    attr_true : TreeNode
        Root of subtree in right (i.e. value of node attribute is 1) branch.

    """
    def __init__(self, value=0, attr_false=None, attr_true=None):
        self.value = value
        self.attr_false = attr_false
        self.attr_true = attr_true
    
def entropy(target):
    """
    Computes entropy of a dataset given its outcomes

    Parameters
    ----------
    target : numpy.ndarray
        Outcomes for each case in a certain dataset.

    Returns
    -------
    entropy : float
        Entropy of the dataset whose outcomes are represented by target.

    """
    
    # Count occurrences of each outcome
    values, count = np.unique(target, return_counts=True)
    
    # Compute probability for each outcome
    p_i = count/len(target)
    
    # 0 yields nan when computing log2
    # Replace with 1. Since log2(1) = 0,
    # 0*log2(0) = 0 is kept
    p_i = np.ma.masked_equal(p_i, 0).filled(1)
    
    entropy = np.sum(-p_i*np.log2(p_i))
        
    return entropy
        
def information_gain(data, target):
    """
    Determines the feature that best represents a dataset by
    computing the information gain for everyone of them and finding the 
    maximum value.

    Parameters
    ----------
    data : numpy.ndarray
        Dataset.
    target : numpy.ndarray
        Outcomes for each element of dataset.

    Returns
    -------
    int
        Index of the attribute (i.e. column index from data) with the
        highest information gain.

    """
    
    # Entropy of the dataset
    entropy_s = entropy(target)
    
    # Initialise entropy for each feature
    entropy_a = np.zeros(data.shape[1])
    
    # For each feature:
    for i in range(data.shape[1]):
        # Count occurences of each possible value of the feature
        feature_values, feature_count = np.unique(data[:, i], return_counts=True)
        
        sum_entropy_v = 0
        # For each value of the feature
        for j in range(len(feature_count)):
            # Get target values of cases where current attribute has current value
            target_v = np.ma.masked_where(data[:, i] != feature_values[j], target)
            target_v = target_v.compressed()
            
            # Get entropy for that subset of cases
            entropy_v = entropy(target_v)
            
            # Get probability of current attribute value
            p_v = feature_count[j]/data.shape[0]
          
            # Accumulate entropy of current attribute
            sum_entropy_v += p_v * entropy_v
                                
        # Compute entropy for current attribute
        entropy_a[i] = entropy_s - sum_entropy_v
    
    # Return index of attribute with highest information gain
    return np.argmax(entropy_a)


###################################################
# References

# Quinlan, J. R. (1986). Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81–106

# Quinlan, J. R. (1987a). Simplifying decision trees.
# In International Journal of Man-Machine Studies (Vol. 27, Issue 3, pp. 221–234).
# Elsevier BV. https://doi.org/10.1016/s0020-7373(87)80053-6

# Quinlan, J. R. (1987b). Generating production rules from decision trees
# Proceedings of the 10th International Joint Conference on Artificial Intelligence - Volume 1, pp 304–307

# Ross Quinlan, J. (1992). C4.5: Programs for Machine Learning. Morgan Kaufmann.


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
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    











