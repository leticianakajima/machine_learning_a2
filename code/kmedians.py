import numpy as np
from utils import euclidean_dist_squared
import pdb

class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        medians = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            medians[kk] = X[i]

        M = medians.shape[0]

        while True:
            y_old = np.copy(y)

            # Compute L1 norm to each median
            for obj in range(N):
                lowestDistance = np.inf
                label = np.inf

                for m in range(M):
                    distance = abs(medians[m][0] - X[obj][0]) + abs(medians[m][1] - X[obj][1])
                    if (distance < lowestDistance):
                        lowestDistance = distance
                        label = m

                y[obj] = label

            # Update means
            for kk in range(self.k):
                medians[kk] = np.median(X[y==kk], axis=0)

            medians[np.isnan(medians)] = np.inf
            changes = np.sum(y != y_old)
           #  print('Running K-medians, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.medians = medians

    def predict(self, X):
        medians = self.medians
        N = X.shape[0]
        M = medians.shape[0]
        yhat = np.ones(N)

        for obj in range(N):
            lowestDistance = np.inf
            label = np.inf

            for m in range(M):
                distance = abs(medians[m][0] - X[obj][0]) + abs(medians[m][1] - X[obj][1])
                if (distance < lowestDistance):
                    lowestDistance = distance
                    label = m

            yhat[obj] = label

        return yhat

    def error(self, X):
        medians = self.medians
        N = X.shape[0]
        M = medians.shape[0]
        distances = np.ones(N)

        for obj in range(N):
            lowestDistance = np.inf
            label = np.inf

            for m in range(M):
                distance = abs(medians[m][0] - X[obj][0]) + abs(medians[m][1] - X[obj][1])
                if (distance < lowestDistance):
                    lowestDistance = distance
                    label = m

            distances[obj] = lowestDistance

        return np.sum(distances)
