# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.io import imread, imshow, imsave


# our code
from naive_bayes import NaiveBayes

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from kmedians import Kmedians
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN

# pdb
import pdb

def plot_2dclustering(X,y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Cluster Plot')


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        pdb.set_trace()


    elif question == '1.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Random Forest (sklearn) validation error: %.3f" % v_error)

        model = NaiveBayes(num_classes=4, beta=1)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error)

    elif question == '2':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)


        print("Our implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        #raise NotImplementedError()
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))

        print("sklearn implementations")
        print("  Decision tree info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain, more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))


    elif question == '3':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        print("Error:", model.error(X))
        plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '3.1':
        X = load_dataset('clusterData.pkl')['X']
        N = 50
        lowestError = np.inf

        model = Kmeans(k=4)
        model.fit(X)

        # for n in range(N):
        #     model.fit(X)
        #     if (model.error(X) < lowestError):
        #         lowestError = model.error(X);
        #         plot_2dclustering(X, model.predict(X))
        #
        # fname = os.path.join("..", "figs", "kmeans_3.1.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)
        # print("\nError: ", lowestError)

    elif question == '3.2':
        X = load_dataset('clusterData.pkl')['X']
        N = 50
        K = 10
        lowestErrors = np.empty([K, 1]) # generate 10 spaces for 1-10

        # generate numbers from 1 up to, but not including 11
        for k in range(1, K + 1):
            model = Kmeans(k=k)
            lowestError = np.inf
            for n in range(N):
                model.fit(X)
                if (model.error(X) < lowestError):
                    lowestError = model.error(X)

            # put numbers into 0 indexed array, 0-9
            np.put(lowestErrors, k - 1, lowestError)

        # create a plot from 1 up to, but not including 11
        plt.plot(range(1, K + 1), lowestErrors)
        plt.ylabel('minimum error')
        plt.xlabel('k')

        fname = os.path.join("..", "figs", "kmeans_3.2.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '3.3':
        X = load_dataset('clusterData2.pkl')['X']
        N = 50
        K = 10

        lowestErrors = np.empty([K, 1])

        for k in range(1, K + 1):
            model = Kmedians(k=k)
            lowestError = np.inf
            for n in range(N):
                model.fit(X)
                if (model.error(X) < lowestError):
                    lowestError = model.error(X)

            np.put(lowestErrors, k - 1, lowestError)

        plt.plot(range(1, K + 1), lowestErrors)
        plt.ylabel('minimum error')
        plt.xlabel('k')

        fname = os.path.join("..", "figs", "kmeans_3.3_medians_graph.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '3.4':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=30, min_samples=2)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plot_2dclustering(X,y)
        fname = os.path.join("..", "figs", "clusterdata_dbscan.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '4':
        img = imread(os.path.join("..", "data", "mandrill.jpg"))

        # part 1: implement quantize_image.py
        # part 2: use it on the doge
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img)
            d_img = quantizer.dequantize(q_img)

            plt.figure()
            plt.imshow(d_img)
            fname = os.path.join("..", "figs", "b_{}_image.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)

            plt.figure()
            plt.imshow(quantizer.colours[None] if b/2!=b//2 else np.reshape(quantizer.colours, (2**(b//2),2**(b//2),3)))
            plt.title("Colours learned")
            plt.xticks([])
            plt.yticks([])
            fname = os.path.join("..", "figs", "b_{}_colours.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)


    else:
        print("Unknown question: %s" % question)
