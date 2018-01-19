import numpy as np


"""
This is where the actual work is done. 
"""


# SELECTORS

class AttributeSelector:
    """
    AttributeSelectors allow trees to opt for different approaches to getting 
    good attributes and attribute values to split on.
    """
    def __init__(self, attribute_selection_type="standard decision tree"):
        switcher = {"standard decision tree": decision_tree_selection, "subset decision": subset_selection_log2}
        assert attribute_selection_type in switcher.keys(), "attribute selection type invalid"
        self.attr_sel = switcher[attribute_selection_type]
        self.attribute_selection_type = attribute_selection_type
        
    def get_attribute_selector(self):
        return self.attr_sel

    def get_selector_name(self):
        return self.attribute_selection_type



def decision_tree_selection(X, Y):
    """
    The decision tree selection method is a slower method which is deterministic 
    and always selects the attribute which is "optimal" to split on.
    Runtime is in O(nr_attributes*nr_examples)

    Keyword arguments:
    X : numpy.ndarray float 2d
    Y : numpy ndarray float 1d

    Returns:
    best_attr_idx, best_attr_val, best_info_gain
    """
    nr_examples, nr_attributes = X.shape[0], X.shape[1]
    attribute_information_gains = np.zeros(nr_attributes)
    attribute_values = np.zeros(nr_attributes)
    for attr_idx in range(nr_attributes):
        col = X[:, attr_idx]
        unique_vals = np.unique(col) # also sorts
        possible_split_values = unique_vals[:-1] + 0.5*(unique_vals[1:]-unique_vals[:-1])
        for split_value in possible_split_values:
            curr_information_gain = information_gain(X,Y, attr_idx, split_value)
            if curr_information_gain>attribute_information_gains[attr_idx]:
                attribute_information_gains[attr_idx] = curr_information_gain
                attribute_values[attr_idx] = split_value
    best_attr_idx  = np.argmax(attribute_information_gains)
    best_attr_val  = attribute_values[best_attr_idx]
    best_info_gain = attribute_information_gains[best_attr_idx]
    return best_attr_idx, best_attr_val, best_info_gain



def subset_selection_log2(X, Y):
    """
    This is a faster method which selects a good attribute to split on but 
    randomly.
    Runtime is in O(log2(nr_attributes)*nr_examples)

    Keyword arguments:
    X : numpy.ndarray float 2d
    Y : numpy ndarray float 1d

    Returns:
    best_attr_idx, best_attr_val, best_info_gain
    """
    nr_examples, nr_attributes = X.shape[0], X.shape[1]
    attribute_information_gains = np.zeros(nr_attributes)
    attribute_values = np.zeros(nr_attributes)
    rand_subset = np.random.choice(nr_attributes, int(np.log2(nr_attributes)), replace=False)
    for attr_idx in rand_subset:
        col = X[:, attr_idx]
        unique_vals = np.unique(col) # also sorts
        possible_split_values = unique_vals[:-1] + 0.5*(unique_vals[1:]-unique_vals[:-1])
        for split_value in possible_split_values:
            curr_information_gain = information_gain(X,Y, attr_idx, split_value)
            if curr_information_gain>attribute_information_gains[attr_idx]:
                attribute_information_gains[attr_idx] = curr_information_gain
                attribute_values[attr_idx] = split_value
    best_attr_idx  = np.argmax(attribute_information_gains)
    best_attr_val  = attribute_values[best_attr_idx]
    best_info_gain = attribute_information_gains[best_attr_idx]
    #if best_info_gain<0.0000001: print "weird gain: ", best_attr_idx, best_attr_val, best_info_gain
    return best_attr_idx, best_attr_val, best_info_gain



# METRICS FOR EVALUATION OF SPLIT VALUE

def information_gain(X, Y, attribute_idx, attribute_value, gini=False):
    """
    Computes the information gain as described here:
    https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
    """
    idx_smaller = X[:, attribute_idx] <= attribute_value
    len_smaller = 1.0*sum(idx_smaller)
    idx_larger  = X[:, attribute_idx] >  attribute_value
    len_larger  = 1.0*len(X) - len_smaller
    H_all = gini_impurity(Y)               if gini else entropy(Y)
    H_sma = gini_impurity(Y[idx_smaller]) if gini else entropy(Y[idx_smaller])
    H_lar = gini_impurity(Y[idx_larger])  if gini else entropy(Y[idx_larger])
    return H_all - (len_smaller/len(X)*H_sma + len_larger/len(X)*H_lar)

def entropy(Y):
    """
    Returns the entropy as defined in information theory, described here:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    entropy = 0
    u, indices = np.unique(Y, return_inverse=True)
    bincount = np.bincount(indices)
    probs_of_classes = bincount/(1.0*len(Y))
    for p in probs_of_classes:
        entropy -= p * np.log2(p)
    return entropy


def gini_impurity(Y):
    """
    Not really used. Just implemented for completeness.
    """
    gini = 0
    u, indices = np.unique(Y, return_inverse=True)
    bincount = np.bincount(indices)
    probs_of_classes = bincount/(1.0*len(Y))
    for val in probs_of_classes:
        gini += val*(1-val)
    return gini



if __name__=="__main__":
    pass








