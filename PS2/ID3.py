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

    classDis = getClassDistribution(examples)
    if len(examples) == 0:
        leaf = Node()
        leaf.setLabel(default)
    elif classDis[0][1] == len(examples):
        leaf = Node()
        leaf.setLabel(classDis[0][0])
        return leaf
    else:
        bestAt = bestAttribute(examples)
        newNode = Node()
        newNode.setCondition(bestAt)


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

# Use information gain to find the best attribute to split on for this set of examples
def bestAttribute(examples):


# Returns two arrays of the example set.
# First array will contain examples that meet the attribute requirement
# Second array will contain examples that do NOT meet the attribute requirement
# Attribute will be a dictionary that describes the condition needed
def sortExamplesByAttribute(examples, attribute):




# Return list of class distributions. First one in list is the mode
def getClassDistribution(examples):
    classList = []
    for i in range(0, len(examples) - 1):
        classList[i] = examples[i].get('Class')
    data = Counter(classList)
    return data.most_common()