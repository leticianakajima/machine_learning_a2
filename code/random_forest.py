import numpy as np
import utils
from random_tree import RandomTree
from random_stump import RandomStumpInfoGain
from scipy import stats
from decision_tree import DecisionTree


class RandomForest(RandomTree):

    #makes 1 randomTree + num_tree parameter to self
    def __init__(self, max_depth, num_trees):
        RandomTree.__init__(self,max_depth)
        self.num_trees = num_trees

    #makes num_tree randomTrees
    def fit(self,X,y):
        self.Random_Forest = []
        for x in range(0, self.num_trees):
            New_tree = RandomTree(self.max_depth)
            New_tree.fit(X,y)
            self.Random_Forest.append(New_tree)
            #add random tree to array of random trees called Random_Forest

    #predict normally and take the mode

    def predict(self, X):
        self.array_preds = []
        for Random_Tree in self.Random_Forest:
            self.array_preds.append(Random_Tree.predict(X))
        return (stats.mode(self.array_preds)[0][0])
