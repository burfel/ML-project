import numpy as np
import time

"""
DecisionNode is the class that builds the tree, top to bottom. It's a recursive
data structure which gets data assigned to itself and checks whether it is a 
leaf given its data. If the node is it terminates. If it is not then it divides
the data into two subsets given its trees techniques and recursively passes on 
the data to two child nodes
"""


class DecisionNode:

    def __init__(self, parent_tree, X, Y, depth, view_progress=False):
        self.d = depth
        self.node_class = None
        if view_progress: print("\nnew node\n", "data size: ", len(X))


        # check whether this is a leaf
        ld = parent_tree.ld
        if ld.is_leaf(depth, X, Y):
            self.node_class = ld.get_leaf_class(Y)
            return

        # selects the attribute as well as the value to split it on
        attrs = parent_tree.attrs
        self.attribute, self.separation_value, self.information_gain = attrs(X, Y)
        
        # split the data so that the children get different parts
        left_idx  = X[:,self.attribute] <= self.separation_value
        right_idx = X[:,self.attribute] >  self.separation_value
        
        self.lc = DecisionNode(parent_tree, X[left_idx],  Y[left_idx],  depth+1)
        self.rc = DecisionNode(parent_tree, X[right_idx], Y[right_idx], depth+1)

    def classify(self,example):
        """
        Trees classify but the actual classification procedure must run the example
        which is being classified by the internal node structure. Doing this requires
        the nodes being capable of classification.
        """
        if self.node_class != None: return self.node_class
        if example[self.attribute] <= self.separation_value:
            return self.lc.classify(example)
        return self.rc.classify(example)

    # helper function
    def print_me(self, name=""):
        if self.node_class==None:
            if self.lc==None or self.rc==None: return # random subset creates occasional dead nodes
            self.lc.print_me(name+"l")
            self.rc.print_me(name+"r")
        else:
            print(name, self.node_class)







