from preprocessing import holdout
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

def predict(model, xTrain, yTrain, xTest, yTest):
    # Test
    # Accuracy
    yHatTest = model.predict(xTest)
    testAcc = metrics.accuracy_score(yTest, yHatTest)

    # predict training and testing probabilties
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                            yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)

    test_precision = metrics.precision_score(yTest, yHatTest[:,1].astype(int)) 
    test_recall = metrics.recall_score(yTest, yHatTest[:,1].astype(int))
    test_matrix = metrics.confusion_matrix(yTest, yHatTest[:,1].astype(int)).ravel()

    # Train
    # Accuracy
    yHatTrain = model.predict(xTrain)
    trainAcc = metrics.accuracy_score(yTrain, yHatTrain)

    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                            yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)

    train_precision = metrics.precision_score(yTrain, yHatTrain[:,1].astype(int))
    train_recall = metrics.recall_score(yTrain, yHatTrain[:,1].astype(int))
    train_matrix = metrics.confusion_matrix(yTrain, yHatTrain[:,1].astype(int)).ravel()

    result_metrics = {
        "trainAcc": trainAcc, 
        "trainAuc": trainAuc, 
        "testAcc": testAcc, 
        "testAuc": testAuc,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_(tn, fp, fn, tp)": train_matrix,
        "test_(tn, fp, fn, tp)": test_matrix
        }

    return result_metrics

def train(xTrain, yTrain, model_name="knn", grid_search=False):
    model = None # Actual model to return

    # K-nearest neighbors
    if model_name == "knn":
        if grid_search:
            model = GridSearchCV(
                KNeighborsClassifier(), 
                [{'n_neighbors': range(1,10,2), 'metric': ['euclidean', 'manhattan']}], cv=2, scoring='f1_macro', verbose=3)
        else:
            model = KNeighborsClassifier()
        
    # Decision tree
    elif model_name == "dt":
        pass

    # Gradient Descent Boosted Decision Tree (GDBDT)
    elif model_name == "GDBDT":
        pass

    # Graph Convulotional Network (GCN)
    elif model_name == "GCN":
        pass

    # Random Forest Classifier
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        if grid_search:
            model = GridSearchCV(
                RandomForestClassifier(),
                [{'criterion': ['gini', 'entropy'], 'max_depth': range(1,32,2)}], cv=2, scoring='f1_macro', verbose=3)
        else:
            model = RandomForestClassifier()

    # Multilayer Perceptron
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        if grid_search:

            model = GridSearchCV(
                MLPClassifier(),
                [{'learning_rate_init':[0.001], 'learning_rate':['invscaling', 'adaptive'], 'batch_size':[1, 4]}], cv=2, scoring='f1_macro', verbose=3)
        else:
            model = MLPClassifier()

    # Fit model before returning it
    model.fit(xTrain, yTrain)

    # Return trained model
    return model



def main():

    # Read data
    train_data = pd.read_csv("./data/binary_train.csv")
    test_data = pd.read_csv("./data/binary_test.csv")

    yTrain = train_data['LABEL'].copy()
    xTrain = train_data.copy()
    xTrain.drop(columns=['LABEL'], inplace=True)

    yTest = test_data['LABEL'].copy()
    xTest = test_data.copy()
    xTest.drop(columns=['LABEL'], inplace=True)

    # Initialize and train model
    model = train(xTrain, yTrain, "mlp", grid_search=True)

    # Predict
    results = predict(model, xTrain, yTrain, xTest, yTest)
    
    # If grid search, prints best parameters chosen
    try:
        if model.best_params_:
            print("Best parameters: ", model.best_params_)
    except:
        print("Error no best params")


    for key, value in results.items():
        print(f"{key}: {value}")




if __name__ == "__main__":
    main()
