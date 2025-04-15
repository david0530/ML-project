import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Knn(object):
    def __init__(self, k):
        self.k = k
        self.isFitted = False

    def train(self, xFeat, y):
        self.xFeat = xFeat
        self.y = y
        self.nFeatures = xFeat.shape[1]
        self.nSamples = xFeat.shape[0]
        self.isFitted = True
        return self

    def predict(self, xFeat):
        yHat = np.array([])
        for sample in xFeat:
            distances = np.linalg.norm(self.xFeat - sample, axis=1)
            labelDistances = [(dist, self.y[i]) for i, dist in enumerate(distances)]
            labelDistances.sort(key=lambda x: x[0])
            kLabels = np.array(labelDistances)[:self.k, 1]
            kVote = stats.mode(kLabels, keepdims=False).mode
            yHat = np.append(yHat, kVote)
        return yHat

def accuracy(yHat, yTrue):
    return np.mean(yHat == yTrue)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain", default="train_features.csv")
    parser.add_argument("--yTrain", default="train_labels.csv")
    parser.add_argument("--xTest", default="test_features.csv")
    parser.add_argument("--yTest", default="test_labels.csv")
    parser.add_argument("--maxK", type=int, default=20, help="maximum k to try")
    args = parser.parse_args()

# Load data
    xTrain_df = pd.read_csv(args.xTrain)
    xTrain = xTrain_df.drop(columns=xTrain_df.columns[0]).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()

    xTest_df = pd.read_csv(args.xTest)
    xTest = xTest_df.drop(columns=xTest_df.columns[0]).dropna().to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()


    k_values = list(range(1, args.maxK + 1))
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        knn = Knn(k)
        knn.train(xTrain, yTrain)

        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain)

        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest)

        train_accuracies.append(trainAcc)
        test_accuracies.append(testAcc)

        print(f"k = {k}: Train Acc = {trainAcc:.3f}, Test Acc = {testAcc:.3f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='s')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy for Different k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
