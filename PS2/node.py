from scipy.stats import mode
from collections import Counter

class Node:
    def __init__(self):
        self.label = None # Class Label
        self.children = {} # If children is empty OR all examples are of the same class, it is a leaf node
        self.attribute = None # Hold the attribute used to compare/split with

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label = label

    def getChildren(self):
        return self.label

    def addChildren(self, key, value):
        self.children[key] = value

    def getAttribute(self):
        return self.condition

    def setAttribute(self, attribute):
        self.attribute = attribute



    # # Return list of class distributions. First one in list is the mode
    # def getClassDistribution(self):
    #     classList = []
    #     for i in range(0, len(self.children) - 1):
    #         classList[i] = self.children[i].get('Class')
    #     data = Counter(classList)
    #     return data.most_common()
    #
    # def getExampleAttributes(self):


	# you may want to add additional fields here...