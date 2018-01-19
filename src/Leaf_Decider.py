import numpy as np

"""
LeafDecider is passed to a classifier which gives nodes access to it. The nodes
use this leaf decider to understand whether they are leaves during their 
creation step. 
If they are leaves then the LeafDecider can compute their class
"""


class LeafDecider:
    def __init__(self, maximum_depth=50, min_examples_per_leaf=1):
        assert type(maximum_depth) is int,            "maximum depth is not of type int"
        assert maximum_depth>0,                        "maximum depth is zero or lower"
        assert type(min_examples_per_leaf) is int,    "minimum examples per leaf is of type int"
        assert min_examples_per_leaf>0,                "minimum number of examples in leaf is irrational"

        self.min_ex = min_examples_per_leaf
        self.max_d  = maximum_depth

    def is_leaf(self, depth, X, Y):
        # this leaf consists of one class only
        if depth >= self.max_d:        return True
        if len(Y) <= self.min_ex:    return True
        if np.allclose(Y, Y[0]):    return True
        return not self.is_separable(X)

    def is_separable(self, X):
        return not np.all(X==X[0])

    def get_leaf_class(self, Y): # get most frequent class
        if len(Y)==0: return -1
        u, indices = np.unique(Y, return_inverse=True)
        return u[np.argmax(np.bincount(indices))]

def test_LD_1():
    ld = LeafDecider()
    X = np.ones(10).reshape(5,2)
    assert not ld.is_separable(X), "test_LD_1 doesn't execute correctly"

def test_LD_2():
    ld = LeafDecider()
    X = np.arange(10).reshape(5,2)
    assert ld.is_separable(X), "test_LD_2 doesn't execute correctly"

def test_LD_3():
    ld = LeafDecider(120, 33)
    assert ld.min_ex==33, "test_LD_3 doesn't execute correctly"
    assert ld.max_d==120, "test_LD_3 doesn't execute correctly"

def test_LD_4():
    ld = LeafDecider()
    Y = np.arange(10) % 3
    assert ld.get_leaf_class(Y)==0, "test_LD_4 doesn't execute correctly"
    Y = np.arange(100)
    Y[10] = 1
    assert ld.get_leaf_class(Y)==1, "test_LD_4 doesn't execute correctly"

if __name__=="__main__":
    # Tests
    
    test_LD_1()
    test_LD_2()
    test_LD_3()
    test_LD_4()







