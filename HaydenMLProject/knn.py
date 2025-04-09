import argparse
import numpy as np
import pandas as pd
from scipy import stats # Slide 3 said we could import this library.


class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?


    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        self.xFeat = xFeat # store training data
        self.y = y
        self.nFeatures = xFeat.shape[1] # Get number of columns (features)
        self.nSamples = xFeat.shape[0] # Number of rows (samples)
        self.isFitted = True

        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape (m, )
            Predicted class label per sample
        """
        yHat = np.array([])
        for sample in xFeat:
            distances = np.linalg.norm(self.xFeat - sample, axis=1) # For training, distances for each should be 0 when its the same node. So with k=1 we should get 100% accuracy.
            labelDistances = [(dist, self.y[i]) for i, dist in enumerate(distances)]
            labelDistances.sort(key=lambda x: x[0]) # O(nlog(n)) complexity
            kLabels = np.array(labelDistances)[:self.k, 1]
            kVote = stats.mode(kLabels).mode
            yHat = np.append(yHat, kVote)
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = np.mean(yHat == yTrue) # Fancy np function which quickly gives the mean which is the accuracy in our case.
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="train_features.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="train_labels.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="test_features.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="test_labels.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain)
    xTrain = xTrain.drop(xTrain.columns[0], axis=1)
    xTrain = xTrain.to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest)
    xTest = xTest.drop(xTest.columns[0], axis=1)
    xTest = xTest.dropna()
    xTest = xTest.to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
