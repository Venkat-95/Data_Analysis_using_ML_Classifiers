import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils.comp_utils import svc_classifier_train, classifier_test, logistic_classifier_train
from utils.vis_utils import visualization_pca, visualization_t_SNE


def preprocessing(_data: pd.DataFrame, stage, _imputer=None, _scalar=None):
    if stage == "Train":
        num_cols = _data.shape[1] - 1
        X = np.empty((0, num_cols), dtype=_data.dtypes[0])
    else:
        num_cols = _data.shape[1]
        X = np.empty((0, num_cols), dtype=_data.dtypes[0])
    y = []

    for index, row in _data.iterrows():
        data_point = row.tolist()
        if data_point[-1] not in [0, 1]:
            continue
        if stage == "Train":
            X = np.concatenate((X, [data_point[:-1]]), axis=0)
            y.append(data_point[-1])
        else:
            X = np.concatenate((X, [data_point]), axis=0)

    if _imputer is not None:
        X = _imputer.transform(X)
    else:
        _imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        X = _imputer.fit_transform(X)

    if _scalar is not None:
        X = _scalar.transform(X)
    else:
        _scalar = StandardScaler()
        X = _scalar.fit_transform(X)

    return X, y, _imputer, _scalar


if __name__ == "__main__":
    data_path = input("Enter the path to data: ")
    train_file = f"{data_path}/data_train.txt"
    data = pd.read_csv(train_file, na_values="?", header=None)
    train_X, train_y, imputer, scalar = preprocessing(data, stage="Train")
    print("Shape of X Train: ", train_X.shape)
    print("Length of y train: ", len(train_y))
    svc_classifer, best_C, best_gamma, best_score = svc_classifier_train(train_X, train_y)
    print(f"Best Score for the validation set: {best_score}")
    logistic_regression_clf, score_logits = logistic_classifier_train(train_X, train_y)
    print(f"Train_Score :{score_logits}")

    test_file = f"{data_path}/data_test.txt"
    data_test = pd.read_csv(test_file, na_values="?", header=None)
    test_X, test_y, imputer, scalar = preprocessing(data_test, stage="Test", _imputer=imputer, _scalar=scalar)
    y_pred_test = classifier_test(clf=svc_classifer, X_test=test_X)
    y_pred_logistic = logistic_regression_clf.predict(test_X)
    visualization_pca(X_test=test_X, y_pred=y_pred_logistic)
    visualization_t_SNE(X_test=test_X, y_pred=y_pred_logistic)
