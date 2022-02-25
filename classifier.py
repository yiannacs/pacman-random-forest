# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
import copy

class Classifier:
    """
    Random forest made up of a settable number of trees (pruned or unprunned).
    If the number of trees is set to 1, it behaves like a standard, 
    single decission tree. 
    
    Attributes
    ----------
    trees : list(DecisionTree)
        List of trees that make up the forest.
    nTrees : int
        Number of trees in the forest.   
    trained : bool
        Indicates whether model has been trained. Used as a failsafe in
        the case that an empty traning set is provided

    """
    
    def __init__(self, nTrees=10):
        print('Creating random forest with {} trees'.format(nTrees))
        self.trees = [DecisionTree() for i in range(nTrees)]
        self.nTrees = nTrees
        self.trained = False

    def reset(self):
        pass
    
    def fit(self, data, target, prune=True):
        """
        Builds tree(s) to make up a random forest.

        Parameters
        ----------
        data : numpy.ndarray
            Dataset cases to train the tree(s).
        target : numpy.ndarray
            Target values for the given cases.
        prune : bool, optional
            Indicates whether to prune the trees in the forest.

        Returns
        -------
        None.

        """
        
        # No data provided
        if len(data) == 0:
            print('Empty training set')
            return
        
        print('Training model')
        data = np.array(data)
        target = np.array(target)
        
        if self.nTrees == 1:
            # Train tree with set as it is
            self.trees[0].fit(data, target)
        else:
            # Rule out features evidently with no effect            
            attributes = []
            # If there's more than one class
            if not np.equal(target, target[0]).all():
                # For each attribute
                for i in range(data.shape[1]):
                    # Attribute has more than one value
                    if not np.equal(data[:,i], data[0, i]).all():
                        # Make feature selectable
                        attributes.append(i)
            attributes = np.array(attributes)
            # Compute number of features to select (Breiman, 2001)
            sizeFeatureSplit = int(np.log2(len(attributes) + 1))
            
            for i in range(self.nTrees):
                # Random feature selection
                treeFeatures = np.random.choice(attributes, sizeFeatureSplit, replace=False)
                
                # Bagging
                # randomly select len(data) cases with replacement
                treeCases = np.random.choice(range(len(data)), len(data))
                treeData = data[treeCases, :]
                treeData = treeData[:, treeFeatures]
                treeTarget = target[treeCases]

                # Breiman (2001) doesn't prune the trees that make up the forest,
                # but I've observed that pruned trees result in less looping
                # in the same couple of cells
                self.trees[i].fit(treeData, treeTarget, prune=prune)
        
        self.trained = True
        print('Done training')
        
    def predict(self, data, legal=None):
        """
        Predicts correct move based on state of the game given by data.
        If the model is not trained, returns a random choice.

        Parameters
        ----------
        data : list(int)
            List of bits describing the state of the game.
        legal : list(int), optional
            List of legal moves. The default is None.

        Returns
        -------
        int 
            Predicted 'good' move based on data.

        """
        
        if self.trained == False:
            print('Model is not trained, returning a random move')
            return np.random.choice(range(4))
            
        data = np.array(data)

        if self.nTrees == 1:    
            # If a single tree, simply return its prediction
            return self.trees[0].predict(data, legal)
        else:
            # If many trees, gather their predictions
            predictions = []
            
            for tree in self.trees:
                predictions.append(tree.predict(data, legal))
                
            # Return the most common prediction
            return np.argmax(np.bincount(predictions))
            
class DecisionTree:
    """
    Decission tree.
    
    Attributes
    ----------
    root : TreeNode
        Root node of the tree.
    default : int
        Class to return by default if a given case does not fit any rule.
    pruned : bool
        Indicates whether the tree has been pruned or not. Used by class methods
        to choose the correct way to predict outcomes
    paths : list(dict('path', 'target', 'cf'))
        Rules table represented by a list.
        Each entry in the list corresponds to one rule.
        Rules are represented as a dictionary with elements 'path'
        (conditions in the rule), 'target' (predicted class for cases that
        follow the rule) and 'cf' (certainty factor or accurary of the rule)
        Rule conditions are represented as a pair of (attributeIndex, value),
        where attributeIndex is the column of the index evaluated by the
        condition, and value is the value the attribute must have to comply
        with the rule.        

    """
    
    def __init__(self):
        self.root = None
        self.default = None
        self.pruned = False
        self.paths = None
        
    def fit(self, data, target, prune=True):
        """
        Fit tree to data by doing the following steps. 
            
        1. Build tree
        2. Rule post-prunning 
        3. Sort rules by certainty factor
        4. Select a default class
                
        Tree-building mostly based on ID3.
        Remaining steps based on C4.5 but, for simplicity,
        also previous works by Quinlan.
            
        Parameters
        ----------
        data : numpy.darray
            Training set data
        target : numpy.darray
            Classes for training data
        prune : bool, optional
            If True, converts the tree to a rule set and does rule post-prunning.
            If False, simply fits the tree to the given data.
            The default is True.

        Returns
        -------
        None.

        """
        
        # Convert data to numpy array
        # data = np.array(data)
        # target = np.array(target)
        
        # Not splitting data
        dataTrain = data
        targetTrain = target
        
        # Add headers to training data:
        # [0, 1, 2, ..., numberOfAttributes - 1]
        # This makes it easier to keep indexes when splitting/taking attributes out
        idx = np.array(range(data.shape[1])).reshape(1, data.shape[1])
        dataTrain = np.concatenate((idx, dataTrain), axis=0)
        
        # Build tree
        self.root = self.buildTree(dataTrain, targetTrain)
        
        if prune == True:
            # Generate rule table            
            self.paths = self.generateRuleTable(self.root)
            
            # Rule post-prunning
            # Pass data without attribute headers
            self.paths = self.rulePostPrunning(self.paths, dataTrain[1:len(dataTrain), :], targetTrain)
            
            # Sort by certainty factor (Quinlan, 1987a)
            self.paths.sort(reverse=True, key=lambda x: x['cf'])
            
            # Find default path  
            self.default = self.findDefault(dataTrain[1:len(dataTrain), :], target)
            
            self.pruned = True
        else:
            self.pruned = False
        
    def generateRuleTable(self, node):
        """
        Generates rule table for tree rooted at node

        Parameters
        ----------
        node : TreeNode
            Root of the tree from which to generate rules.

        Returns
        -------
        paths : list(dict('path', 'target'))
            List of rules.
            Each entry is a dictionary containing 'path', which in turn
            is a list of pairs (attributeIndex, value), and 'target', which
            is the class the rule in 'path' leads to.

        """
        
        paths = []
        
        path = []
        self.traverse(node, paths, list(path))
        self.traverse(node, paths, list(path))
        
        return paths
    
    def predict(self, data, legal=None):
        """
        Predict good move given state defined by data.
        If the predicted 'good' move is not legal, returns default move.

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
        
        if self.pruned == True:
            # Predict using simplified rules table
            move = self.predictWithRules(data)
        else:
            move = self.traverseWithData(self.root, data)
            
        if move not in legal:
            return self.default
        
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
        
        # All remaining test cases are equal
        # Excluding first row from check as those are headers
        if np.equal(data[1:,:], data[1]).all():
            # Return most common class
            return TreeNode(value=np.argmax(np.bincount(target)))

        # Find the best attribute left
        # All features are binary, so information gain is a good metric
        # to determine importance of features.
        # Need only the index in the current array, so don't need to pass headers
        attrMaxGainCurrentIndex = informationGain(data[1:, :], target)
        
        # Divide data by values of the attribute with greatest gain
        x0, y0, x1, y1 = self.branchOutAttribute(attrMaxGainCurrentIndex,
                                                 data,
                                                 target)
        
        # No cases when attribute == 0
        if len(y0) == 0:
            # Set branch to leaf with class given by the most common class at this node
            left = TreeNode(value=np.argmax(np.bincount(y0)))
        # Still have data,
        else:
            # recursively built this branch
            left = self.buildTree(x0, y0)

        # Determine if left branch points to leaf            
        if left.attrFalse is None and left.attrTrue is None:
            leftIsLeaf = True
        else:
            leftIsLeaf = False

        # No cases with attribute == 0
        if len(y1) == 0:
            # Set branch to leaf with class given by the most common class at this node
            right = TreeNode(value=np.argmax(np.bincount(target)))
        # There are cases where attribute is 1
        else:
            # Recursively build right branch
            right = self.buildTree(x1, y1)

        # Determine if right branch points to a leaf 
        if right.attrFalse is None and right.attrTrue is None:
            rightIsLeaf = True
        else:
            rightIsLeaf = False
        
        # If both branches point to leaves,
        # and both leaves have the same class,
        if rightIsLeaf and leftIsLeaf:
            if right.value == left.value:
                # Discard leaves and set this node to said class
                return TreeNode(value=right.value)
            
        # Note a leaf, return node with branch pointing to subtrees
        return TreeNode(value=data[0][attrMaxGainCurrentIndex], attrFalse=left, attrTrue=right)

    def rulePostPrunning(self, paths, data, target):
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
        
        # Remove rules
        for i in range(len(paths)):
            for step in paths[i]['path']:
                # Deep copy to keep original intact when removing preconditions
                altPath = copy.deepcopy(paths)
                
                # Remove current precondition from rule
                altPath[i]['path'].remove(step)
                
                # Build contingency table
                xy = np.concatenate((data, target.reshape(1, len(target)).T), axis=1)
                
                # Separate cases that belong and not belong to the rule's class
                belongToClass = np.array([xy[l, :] for l in range(xy.shape[0]) if xy[l, -1]==paths[i]['target']])
                notBelongToClass = np.array([xy[l, :] for l in range(xy.shape[0]) if xy[l, -1] != paths[i]['target']])
                
                # Count combinations of [not] belonging to the class and [not] satisfying the rule
                belongSatisfy = np.sum(belongToClass[:, step[0]])
                belongNotSatisfy = belongToClass.shape[0] - belongSatisfy
                notBelongSatisfy = np.sum(notBelongToClass[:, step[0]])
                notBelongNotSatisfy = notBelongToClass.shape[0] - notBelongSatisfy
                
                # Compute certainty factor for original and alternate rule
                certainty = self.computeCertainty(belongSatisfy, notBelongSatisfy)
                altCertainty = self.computeCertainty(belongSatisfy + belongNotSatisfy, notBelongSatisfy + notBelongNotSatisfy)
                
                # Set the rule's certainty factor to the greater one
                paths[i]['cf'] = max([certainty, altCertainty])
                
                # Keep the rule with the greater certainty factor
                if altCertainty >= certainty:
                    paths[i]['path'].remove(step)
        
        return paths

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
        unpredictedTargets = target[np.argwhere(predictions==None)]
        
        # If all cases had a rule, return most common class
        if len(unpredictedTargets) < 1:
            return np.argmax(np.bincount(target))
        
        # Return most common class among unpredicted cases
        return np.argmax(np.bincount(unpredictedTargets.flatten()))
        
    def discardRuleSupersets(self, paths, target):
        """
        For each possible class, remove rules that are supersets of simpler rules.
        
        Currently not used as rules are sorted by certainty factor and this
        could remove a rule with a greater value.

        Parameters
        ----------
        target : numpy.ndarray
            Target from which to remove supersets

        Returns
        -------
        list(dict('path', 'target'))
            Rule list with supersets that lead to target removed

        """
        
        iFoundTarget = False
        for iPath in paths:
            if iPath['target'] == target:
                iFoundTarget = True
                
                # Eat up overhead of looping from the start to not deal with
                # updating indices
                jFoundTarget = False
                for jPath in paths:
                    if jPath['target'] == target:
                        jFoundTarget = True
                
                        # Check if any of pair is subset of another,
                        # delete superset
                        if iPath != jPath:
                            if set(iPath['path']).issubset(jPath['path']):
                                paths.remove(jPath)
                                
                            if set(jPath['path']).issubset(iPath['path']):
                                paths.remove(iPath)
                                
                    elif jFoundTarget:
                        break
            elif iFoundTarget:
                break
            
        return paths
                    
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
            prediction = self.predictWithRules(row, alt)
            predictions.append(prediction)
            
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
        if node.attrTrue is not None:
            # Take left path
            leftPath = list(path)
            leftPath.append((node.value, 0))
            self.traverse(node.attrFalse, paths, leftPath)

            # Take right path
            rightPath = list(path)
            rightPath.append((node.value, 1))                
            self.traverse(node.attrTrue, paths, rightPath)
            
        # Node is a leaf
        if node.attrTrue is None:
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
        if (node.attrFalse is not None) or (node.attrTrue is not None):
            # Print current node as an attribute index
            print('{}\tindex = {}'.format(depth, node.value))
            
            # Keep traversing tree
            self.printTree(node.attrFalse, depth + 1)
            self.printTree(node.attrTrue, depth + 1)
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
        x0 : numpy.ndarray
            Cases data where attribute is 0.
        y0: numpy.ndarray
            Classes corresponding to cases where attribute is 0.
        x1 : numpy.ndarray
            Cases data where attribute is 1.
        y1 : numpy.ndarray
            Classes corresponding to cases where attribute is 1.

        """
        
        x0 = []
        x1 = []
        y0 = []
        y1 = []
        
        # Add headers to both x
        x0.append(data[0])
        x1.append(data[0])
        
        # Divide depending on value of attr
        for i in range(1, data.shape[0]):
            if data[i][attr] == 0:
                x0.append(data[i])
                y0.append(target[i - 1])
            else:
                x1.append(data[i])
                y1.append(target[i - 1])
        
        # Convert to numpy.ndarray
        x0 = np.array(x0)
        x1 = np.array(x1)
        y0 = np.array(y0)
        y1 = np.array(y1)
        
        # Remove attribute from data
        x0 = np.delete(x0, attr, axis=1)
        x1 = np.delete(x1, attr, axis=1)
    
        return x0, y0, x1, y1

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
        for datum in data:
            moves.append(self.predict(datum))
            
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
        if node.attrFalse is None and node.attrTrue is None:
            return node.value
        
        # Node is not a leaf
        attribute = node.value
        # If value in data of node attribute is 1,
        if data[attribute]:
            # Traverse right branch
            return self.traverseWithData(node.attrTrue, data)
        # If value in data of node attribute is 0,
        else:
            # Traverse left branch
            return self.traverseWithData(node.attrFalse, data)
      
    def printRuleTable(self):
        """
        Prints the tree's rule table

        Returns
        -------
        None.

        """
        
        for path in self.paths:
            print(path)
            
        print('Default class: {}'.format(self.default))
        
class TreeNode():
    """
    Decission tree node.
    
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
    attrFalse : TreeNode
        Root of subtree in left (i.e. value of node attribute is 0) branch.
    attrTrue : TreeNode
        Root of subtree in right (i.e. value of node attribute is 1) branch.

    """
    def __init__(self, value=0, attrFalse=None, attrTrue=None):
        self.value = value
        self.attrFalse = attrFalse
        self.attrTrue = attrTrue
    
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
    p = count/len(target)
    
    # 0 yields nan when computing log2
    # Replace with 1. Since log2(1) = 0,
    # 0*log2(0) = 0 is kept
    p = np.ma.masked_equal(p, 0).filled(1)
    
    entropy = np.sum(-p*np.log2(p))
        
    return entropy
        
def informationGain(data, target):
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
    setEntropy = entropy(target)
    
    # Initialise entropy for each feature
    featureEntropy = np.zeros(data.shape[1])
    
    # For each feature:
    for i in range(data.shape[1]):
        # Count occurences of each possible value of the feature
        featureValues, featureCount = np.unique(data[:, i], return_counts=True)
        
        valueEntropySum = 0
        # For each value of the feature
        for j in range(len(featureCount)):
            # Get target values of cases where current attribute has current value
            valueTarget = np.ma.masked_where(data[:, i] != featureValues[j], target)
            valueTarget = valueTarget.compressed()
            
            # Get entropy for that subset of cases
            subsetEntropy = entropy(valueTarget)
            
            # Get probability of current attribute value
            valueP = featureCount[j]/data.shape[0]
          
            # Accumulate entropy of current attribute
            valueEntropySum += valueP * subsetEntropy
                                
        # Compute entropy for current attribute
        featureEntropy[i] = setEntropy - valueEntropySum
    
    # Return index of attribute with highest information gain
    return np.argmax(featureEntropy)


"""
References

Quinlan, J. R. (1986). Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81–106
https://dl.acm.org/doi/10.1023/A%3A1022643204877

Quinlan, J. R. (1987a). Simplifying decision trees.
In International Journal of Man-Machine Studies (Vol. 27, Issue 3, pp. 221–234).
Elsevier BV. https://doi.org/10.1016/s0020-7373(87)80053-6

Quinlan, J. R. (1987b). Generating production rules from decision trees
Proceedings of the 10th International Joint Conference on Artificial Intelligence - Volume 1, pp 304–307
https://dl.acm.org/doi/10.5555/1625015.1625078

Ross Quinlan, J. (1992). C4.5: Programs for Machine Learning. Morgan Kaufmann.
ISBN 1558602380, 9781558602380

Breiman, L. (2001). Random Forests. University of California, Statistics Department
https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
            
"""