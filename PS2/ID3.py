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
    print "Examples"
    print examples
    classDis = getClassDistribution(examples)
    print "Length"
    print len(examples)
    print "first size"
    print classDis[0][1]
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
        atValues = getUniqueAttrValues(examples, bestAt)
        sortedSamples = sortExamplesByAttribute(examples, bestAt, atValues)
        for i in range(len(atValues)):
            print "Values"
            print atValues
            print "i"
            print i
            print "Sorted Examples"
            print sortedSamples[i]
            subtree = ID3_helper(sortedSamples[i], attributes, getClassDistribution(sortedSamples[i]))
            newNode.addChildren(atValues[i], subtree)
        return newNode



def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''


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
        values.remove(values.index('?'))
    values = np.array(values)
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
    print "class distro"
    print data.most_common()
    return data.most_common()

def getUniqueAttrValues(examples, attribute):
    values = []
    for example in examples:
        values.append(example[attribute])
    return np.unique(values)
