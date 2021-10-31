import numpy as np
import pandas as pd
from numpy.random import permutation

def prepare_date():

    # Read files
    train = pd.read_csv("./data/exoTrain.csv")
    test = pd.read_csv("./data/exoTest.csv")


    # Case: Make labels binary & replace order
    train['LABEL'].replace(1, 0, inplace=True)
    test['LABEL'].replace(1, 0, inplace=True)

    train['LABEL'].replace(2, 1, inplace=True)
    test['LABEL'].replace(2, 1, inplace=True)

    # Save new files
    train.to_csv("./data/binary_train.csv", index=False)
    test.to_csv("./data/binary_test.csv", index=False)

def holdout(xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 
    """
    # Shuffle indexes
    p = permutation(len(xFeat))
    xFeat = xFeat.loc[p]
    y = y.loc[p]

    # Second: find split_index from testSize
    split_index = round(testSize*len(y))

    # Split data
    xTrain = xFeat.iloc[0:split_index].reset_index()
    xTest = xFeat.iloc[split_index:len(y)].reset_index()

    yTrain = y.iloc[0:split_index].reset_index()
    yTest = y.iloc[split_index:len(y)].reset_index()

    # Drop index
    xTrain = xTrain.drop(columns="index")
    yTrain = yTrain.drop(columns="index")
    xTest = xTest.drop(columns="index")
    yTest = yTest.drop(columns="index")

    # Compute
    return xTrain, xTest, yTrain, yTest

if __name__ == "__main__":
    # Examples:

    # First step
    prepare_date()

    # Second step
    # y = pd.read_csv("filtered_classes.csv")
    # xFeat = pd.read_csv("filtered_features.csv")

    # xTrain, xTest, yTrain, yTest = holdout(xFeat, y, 0.7)

    # print(xTrain.head())
    # print(yTrain.head())
    # print(xTest.head())
    # print(yTest.head())


