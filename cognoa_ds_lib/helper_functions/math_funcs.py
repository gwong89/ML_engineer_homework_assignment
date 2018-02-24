import math
import itertools
import numpy as np


# converts a value to a float gracefully
def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.NaN

# H(Y) = -sigmaOveri(P(Y=yi)*log2(P(Y=yi)))
def entropy(categoricalVariableAsList):
    probabilityDistributionOfY = pd.Series(categoricalVariableAsList).value_counts(normalize=True)
    entropy = -1.0 * sum([p * math.log(p, 2) for p in probabilityDistributionOfY])
    return entropy


# H(Y|X) = sigmaOverj(P(X=xj)*H(Y|X=xj))
def conditionalEntropy(YasList, XasList):
    pairs = zip(XasList, YasList)
    probabilityDistributionOfX = pd.Series(XasList).value_counts(normalize=True)
    conditionalEntropy = 0.0
    for xj in np.unique(XasList):
        conditionalEntropy += probabilityDistributionOfX[xj] * entropy([item for item in pairs if item[0] == xj])
    return conditionalEntropy


# IG(Y|X) = H(Y) - H(Y|X)
def informationGain(YasList, XasList):
    return entropy(YasList) - conditionalEntropy(YasList, XasList)


def computeCorrelationUsingInfoGain(dataframe, targetName, featureName, missingValueLabel='missing'):
    # filter away rows with missing target of feature values
    df = dataframe[(dataframe[targetName] != missingValueLabel) & (dataframe[featureName] != missingValueLabel)]

    target = df[targetName]
    feature = df[featureName]

    return informationGain(target, feature)

#given a bunch of variables, each in a list of possible ranges, returns tuples for every possible combination of variable values
#   for example, given [ ['a','b'], [1,2], ['x','y'] ]
#   it returns [ ('a',1,'x'), ('a',1,'y'), ('a',2,'x'), ('a',2,'y'), ('b',1,'x') ... ]
def get_combinations(list_of_lists):
    return list(itertools.product(*list_of_lists))

# handy function that operates on a DF column and maps the data range to [0-1]
def normalize_range(column):
    min_val = min(column.values)
    max_val = max(column.values)
    output = (column.values - min_val) / (max_val - min_val)
    return output