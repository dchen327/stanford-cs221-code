#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features = collections.defaultdict(int)
    for word in x.split():
        features[word] += 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    # featureExtractor = extractCharacterFeatures(6) # for problem 3e
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        return 1 if dotProduct(weights, featureExtractor(x)) > 0 else -1

    for x, y in trainExamples:
        for feature in featureExtractor(x):
            weights[feature] = 0
    for i in range(numIters):
        for x, y in trainExamples:
            if dotProduct(weights, featureExtractor(x)) * y < 1:
                increment(weights, eta * y, featureExtractor(x))
        # print(evaluatePredictor(testExamples, predictor))
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {feature: random.random() for feature in random.sample(list(weights), len(weights) - 1)}
        y = 1 if dotProduct(weights, phi) > 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        features = collections.defaultdict(int)
        s = x.replace(' ', '')
        for i in range(len(s)+1-n):
            features[s[i:i+n]] += 1
        return features
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def distance(x, mu):
        """ Return the squared distance between two vectors x and y """
        return sum((x[i] - mu[i])**2 for i in x)

    centers = random.sample(examples, K)
    z = [0] * len(examples)
    for t in range(maxIters):
        # step 1
        for i, x in enumerate(examples):
            min_d = 1000000000
            for k, mu in enumerate(centers):
                d = distance(x, mu)
                if d < min_d:
                    min_d = d
                    z[i] = k
        # step 2
        for k, mu in enumerate(centers):
            sum_x = collections.defaultdict(float)
            count = z.count(k)
            for i, x in enumerate(examples):
                if z[i] == k:
                    increment(sum_x, 1 / count, x)
                centers[k] = sum_x
    # calculate loss
    loss = 0
    for i, x in enumerate(examples):
        diff = x.copy()
        increment(diff, -1, centers[z[i]])
        loss += dotProduct(diff, diff)

    return (centers, z, loss)
    # END_YOUR_CODE

# examples = generateClusteringExamples(2, 4, 2)
# K = 2
# maxIters = 5
# centers, assignments, loss = kmeans(examples, K, maxIters)
# outputClusters('clusters.txt', examples, centers, assignments)