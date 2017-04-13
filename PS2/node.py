from scipy.stats import mode
from collections import Counter

class Node:
    def __init__(self):
        self.label = None # Class Label
        self.children = {} # If children is empty OR all examples are of the same class, it is a leaf node
        self.attribute = None # Hold the attribute used to compare/split with
        self.parent = None # Holds the parent of this Node. Parent is a single Node object. If None, this node is the root of the tree
        # Holds the test samples used to evaluate the split for this node. Will be used to help determine the majority when pruning the tree
        self.testSamples = []

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label = label

    def getChildren(self):
        return self.children

    def setChildren(self, children):
        self.children = children

    def addChildren(self, key, value):
        self.children[key] = value

    def getAttribute(self):
        return self.attribute

    def setAttribute(self, attribute):
        self.attribute = attribute

    def getParent(self, parent):
        self.parent = parent

    def setParent(self, parent):
        return self.parent

    def getTestSamples(self, samples):
        self.testSamples = samples

    def setTestSamples(self):
        return self.testSamples



	# you may want to add additional fields here...