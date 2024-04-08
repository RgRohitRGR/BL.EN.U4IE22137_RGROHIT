import numpy as np
from collections import Counter
from math import log2
import math
import pandas as pd

from sklearn.preprocessing import LabelEncoder

#calculate the entropy of the target variable
def entropy(y):
    n = len(y)
    counts = np.bincount(y)
    probs = counts[np.nonzero(counts)] / n
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

#calculate the gain of the feature
def information_gain(X, y, feature_idx):
    total_entropy = entropy(y)

    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / len(y)) * entropy(y[X[:, feature_idx] == values[i]]) for i in range(len(values))])

    gain = total_entropy - weighted_entropy
    return gain

#finds the root node with the highest information gain
def find_root_node(X, y):
    num_features = X.shape[1]
    gains = [information_gain(X, y, i) for i in range(num_features)]
    print(gains)
    root_feature_idx = np.argmax(gains)
    return root_feature_idx


df = pd.read_excel("C:\\Users\\Rohit\\Downloads\\documents\\data1.xlsx")
df=pd.DataFrame(df)

#merging 2 features in one array as base set
def removeNull(Class):
    iterate=df[Class]
    y=[]
    mean=df[Class].mean(axis=0)
    for i in iterate:
        if(math.isnan(i)):
            y.append(mean)
        else:
            y.append(i)
    y=pd.DataFrame(y)
    df[Class]=y
removeNull('radius_mean')
removeNull('texture_mean')
def labelencode(Class):
    removeNullCategorical(Class)
    temp=df[Class]
    temp=temp.to_numpy()
    temp=temp.flatten()
    LE=LabelEncoder()
    LE.fit(temp)
    array=LE.transform(temp)
    array=pd.DataFrame(array)
    df[Class]=array
    
def isNaN(string):
    return string != string
def removeNullCategorical(Class):
    iterate=df[Class]
    y=[]
    mostfreq=df[Class].value_counts().idxmax()
    for i in iterate:
        if(isNaN(i)):
            y.append(mostfreq)
        else:
            y.append(i)
    y=pd.DataFrame(y)
    df[Class]=y
    return y

removeNullCategorical('diagnosis')
labelencode('diagnosis')
y=df['diagnosis'].to_numpy()
X=df[['radius_mean','texture_mean']].to_numpy()

root_feature_idx = find_root_node(X,y)
print("Root node feature index:", root_feature_idx)


#Q2
import numpy as np
import pandas as pd
from collections import Counter
from math import log2

class DecisionTree:
    def __init__(self):
        self.tree = {}

    def entropy(self, y):
        """
        Calculate the entropy of a target variable.

        Parameters:
        - y: array-like, target variable

        Returns:
        - entropy: float, entropy value
        """
        n = len(y)
        counts = np.bincount(y)
        probs = counts[np.nonzero(counts)] / n
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def information_gain(self, X, y, feature_idx):
        """
        Calculate the information gain of a feature.

        Parameters:
        - X: array-like, feature matrix
        - y: array-like, target variable
        - feature_idx: int, index of the feature to calculate information gain for

        Returns:
        - gain: float, information gain value
        """
        # Calculate total entropy before split
        total_entropy = self.entropy(y)

        # Calculate entropy after split
        values, counts = np.unique(X[:, feature_idx], return_counts=True)
        weighted_entropy = np.sum([(counts[i] / len(y)) * self.entropy(y[X[:, feature_idx] == values[i]]) for i in range(len(values))])

        # Calculate information gain
        gain = total_entropy - weighted_entropy
        return gain

    def find_root_node(self, X, y, binning_type=None, num_bins=None):
        """
        Find the root node feature/attribute with the highest information gain.

        Parameters:
        - X: array-like, feature matrix
        - y: array-like, target variable
        - binning_type: str, type of binning (equal_width or frequency), default=None
        - num_bins: int, number of bins to create, default=None

        Returns:
        - root_feature_idx: int, index of the feature with the highest information gain
        """
        if binning_type and num_bins:
            X = self.binning(X, binning_type, num_bins)
        num_features = X.shape[1]
        gains = [self.information_gain(X, y, i) for i in range(num_features)]
        root_feature_idx = np.argmax(gains)
        return root_feature_idx

    def binning(self, X, binning_type='equal_width', num_bins=10):
        """
        Perform binning on continuous-valued features.

        Parameters:
        - X: array-like, feature matrix
        - binning_type: str, type of binning (equal_width or frequency), default='equal_width'
        - num_bins: int, number of bins to create, default=10

        Returns:
        - X_binned: array-like, binned feature matrix
        """
        X_binned = np.zeros_like(X)
        for i in range(X.shape[1]):
            if binning_type == 'equal_width':
                bins = np.linspace(X[:, i].min(), X[:, i].max(), num_bins + 1)
            elif binning_type == 'frequency':
                bins = np.percentile(X[:, i], np.linspace(0, 100, num_bins + 1))
            else:
                raise ValueError("Invalid binning type. Choose equal_width or frequency.")
            X_binned[:, i] = np.digitize(X[:, i], bins)
        return X_binned

# Read data from Excel file
df = pd.read_excel("C:\\Users\\Rohit\\Downloads\\documents\\data1.xlsx")

# Extract features (X) and target variable (y)
X = df[['radius_mean', 'texture_mean']].values
y = df['diagnosis'].values

# Initialize DecisionTree object
dt = DecisionTree()

# Using default parameters
root_feature_idx = dt.find_root_node(X, y)
print("Root node feature index with default parameters:", root_feature_idx)

# Using equal width binning with 3 bins
root_feature_idx_binned = dt.find_root_node(X, y, binning_type='equal_width', num_bins=3)
print("Root node feature index with equal width binning:", root_feature_idx_binned)

# Using frequency binning with 2 bins
root_feature_idx_freq_binned = dt.find_root_node(X, y, binning_type='frequency', num_bins=2)
print("Root node feature index with frequency binning:", root_feature_idx_freq_binned)
