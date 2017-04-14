from node import Node
from collections import Counter
import numpy as np


# Examples: a list of dictionary. Each dictionary contains information about attributes and the result of classification
#           Assume that all attributes have some sort of value attached to it, include a missing value "?".
#           Assume only discrete values for each attribute, ie 1 or 0, yes or no or ?
# Default: default class. If the example set is empty, return default
def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    atList = getAttributes(examples[0])
    return ID3_helper(examples, atList, default)

# Helper function that will help keep track of attributes that are already split on
def ID3_helper(examples, attributes, default):
    # print "Examples"
    # print examples
    classDis = getClassDistribution(examples)
    # print "Length"
    # print len(examples)
    # print "first size"
    # print classDis[0][1]
    if len(examples) == 0:
        leaf = Node()
        leaf.setLabel(default)
        return leaf
    elif classDis[0][1] == len(examples):
        leaf = Node()
        leaf.setLabel(classDis[0][0])
        return leaf
    elif len(attributes) == 0:
        leaf = Node()
        leaf.setLabel(classDis[0][0])
        return leaf
    else:
        bestAt = bestAttribute(examples, attributes)
        attributes.remove(bestAt)
        newNode = Node()
        newNode.setAttribute(bestAt)
        # print examples
        newNode.setTestSamples(examples)
        atValues = getUniqueAttrValues(examples, bestAt)
        sortedSamples = sortExamplesByAttribute(examples, bestAt, atValues)
        for i in range(len(atValues)):
            # print "Values"
            # print atValues
            # print "i"
            # print i
            # print "Sorted Examples"
            # print sortedSamples[i]
            subtree = ID3_helper(sortedSamples[i], attributes, getClassDistribution(sortedSamples[i]))
            subtree.setParent(newNode)
            newNode.addChildren(atValues[i], subtree)
        return newNode


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    # node: tree; example: val_data
    acc = []
    tree_new = []
    ori_tree = node
    ori_acc = test(node, examples)
    nodeList = search_node(node)
    for n in nodeList:
        tree = ori_tree
        # print "original tree"
        # print breadth_first_search(tree)
        value = getClassDistribution(n.testSamples)[0][0]
        # print "node"
        # print n
        # print "value"
        # print value
        tree = replaceNewTree(tree, n, value)
        tree_new.append(tree)
        # print "new tree"
        # print breadth_first_search(tree)
        acc_new = test(tree, examples)
        acc.append(acc_new)
    max_acc = max(acc)
    # print "Acc list"
    # print acc
    index = acc.index(max_acc)
    if max_acc <= ori_acc:
        tree = ori_tree
    else:
        tree = tree_new[index]

    return tree


def search_node(root):
    s = []
    q = []
    i = 0
    q.append(root)
    while i < len(q):
        current = q[i]
        if len(current.children) != 0:
            s.append(current)
            child = current.children.values()
            for j in range(len(child)):
                q.append(child[j])
        i = i + 1
    return s


def replaceNewTree(tree, node, value):
    parent = node.parent
    # print "parent"
    # print parent
    new_node = creatNewNode(value, node)
    if parent == None:
        return tree
    else:
        for k, v in parent.children.iteritems():
            if v == node:
                parent.children[k] = new_node
        # print "new_node"
        # print new_node
        # print "replace new tree"
        # print breadth_first_search(tree)
        return tree


def creatNewNode(value, node):
    new_node = Node()
    new_node.setAttribute(None)
    new_node.setChildren({})
    new_node.setLabel(value)
    new_node.setParent(node.parent)
    new_node.setTestSamples(node.testSamples)
    return new_node

def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    # print "Test Tree"
    # breadth_first_search(node)
    correct = 0
    for example in examples:
        classVal = evaluate(node, example)
        if classVal == example['Class']:
            correct = correct + 1
    acc = float(correct) / len(examples)
    # print "accuracy"
    # print acc
    return acc


def evaluate(node, example):
  temp = node
  while len(temp.children) != 0:
    na = temp.attribute # na1: first node attribute
    # print na
  # example = dict(a=1, b=0);
    val = example[na]
    # print val

    if val == '?':
        atList = getAttributeDistribution(temp.getTestSamples(), na)
        if atList[0][0] == '?':
            val = atList[1][0]
        else:
            val = atList[0][0]
    # print "val"
    # print val
    # print "children"
    # print temp.children

    if len(temp.children) == 1:
        onlyVal = temp.children.values()
        new_node = onlyVal[0]
    else:
        new_node = temp.children[val]
    temp = new_node
  # print "temp"
  # print temp
  # print "label"
  # print temp.label
  return temp.label

# Returns two arrays of the example set.
# Attribute will be a dictionary that describes the condition needed
def sortExamplesByAttribute(examples, attribute , attribute_values):
    sub_example = []
    for i in range(len(attribute_values)):
        val_example = []
        for example in examples:
            if example[attribute] == attribute_values[i]:
                val_example.append(example)
        sub_example.append(val_example)
    return sub_example

# Takes in one example. Returns a list of attributes in example
# Example is a dictionary containing information about attributes and class
def getAttributes(example):
    attribute = []
    for k in example:
        attribute.extend([k])
    attribute.remove('Class')
    return attribute


# Return lists of different values that attribute can contain
def getAttributeValues(examples, attribute):
    valueList = []
    for i in range(len(examples)):
        valueList = examples[i].get(attribute)
    data = Counter(valueList).most_common()
    for i in range(len(data)):
        valueList[i] = data[i][0]
    return valueList


# Use information gain to find the best attribute to split on for this set of examples
def bestAttribute(examples, attributes):
    maxValue = -1
    maxIndex = -1
    for i in range(len(attributes)):
        infoValue = calcInfoGain(examples, attributes[i])
        if infoValue > maxValue:
            maxValue = infoValue
            maxIndex = i
    return attributes[maxIndex]



# Calculates info gain for specific attribute
def calcInfoGain(examples, attribute):
    values = []
    sub_example = []
    # attr_sample_size = len(examples)?
    # Calculate overall sample set entropy
    Infogain = calcEntropy(examples)

    # Get unique attribute values
    values = getUniqueAttrValues(examples, attribute)
    if '?' in values:
        values.remove('?')
    unique_value = np.unique(values)

    # Sort examples based on attribute values
    attr_sample_size = 0;
    for i in range(len(unique_value)):
        val_example = []
        for example in examples:
            if example[attribute] == unique_value[i]:
                val_example.append(example)
        sub_example.append(val_example)
        attr_sample_size = attr_sample_size + len(val_example)

    # Calculate infogain for attribute
    sub_example = np.array(sub_example)
    for i in range(sub_example.shape[0]):
        entropy = calcEntropy(sub_example[i])
        p = len(sub_example[i]) / attr_sample_size
        Infogain = Infogain - p * entropy

    return Infogain

# Return entropy of data set
def calcEntropy(examples):
    classList = []
    # for i in range(len(examples)):
    #     classList[i] = examples[i].get('Class')
    for example in examples:
        classList.append(example['Class'])
    data = Counter(classList).most_common()

    entropyVal = 0
    sample_size = len(classList)
    for i in range(len(data)):
        majority = data[i][1]
        p1 = majority / float(sample_size)
        entropyVal = entropyVal - p1*np.log2(p1)
    # print majority
    # print minority
    # print sample_size
    # print "P1"
    # print p1
    # print "P2"
    # print p2
    return entropyVal

# Return list of class distributions. First one in list is the mode
def getClassDistribution(examples):
    classList = []
    # for i in range(len(examples)):
    #     classList[i] = examples[i].get('Class')
    for example in examples:
        classList.append(example['Class'])
    data = Counter(classList)
    # print "class distro"
    # print data.most_common()
    return data.most_common()

# Return list of class distributions. First one in list is the mode
def getAttributeDistribution(examples, attribute):
    atList = []
    # for i in range(len(examples)):
    #     classList[i] = examples[i].get('Class')
    for example in examples:
        atList.append(example[attribute])
    data = Counter(atList)
    # print "class distro"
    # print data.most_common()
    return data.most_common()

def getUniqueAttrValues(examples, attribute):
    values = []
    for example in examples:
        values.append(example[attribute])
    values = np.unique(values).tolist()
    if '?' in values:
        values.remove('?')
    return values


def breadth_first_search(root):
    '''
    given the root node, will complete a breadth-first-search on the tree, returning the value of each node in the correct order
    '''
    tree = [root]
    bfsStr = ""
    while len(tree) != 0:
        childList = tree[0].children
        bfsStr += "Label: " + str(tree[0].label) + " " + "Attribute: " + str(tree[0].attribute) + " Key: " + str(tree[0].children.keys()) + " Value: " +  str(tree[0].children.values())  + \
                  " Address: " + str(tree[0]) + "\n"
        if childList != None or len(childList) != 0:
            for k, v in childList.iteritems():
                tree.append(v)
        tree.pop(0)
    return bfsStr
