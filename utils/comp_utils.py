import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn import svm, linear_model


def svc_classifier_train(X_train, y_train):
    C = np.logspace(-3, 3, 6)
    gamma = [0.001, 0.01, 0.1, 1, 100, 1000]
    parameters = {'C': C, 'gamma': gamma}
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.35, random_state=42)

    T_clf = svm.SVC()
    clf = GridSearchCV(estimator=T_clf, param_grid=parameters, n_jobs=4, cv=cv, verbose=10)
    clf.fit(X_train, y_train)

    print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
    best_C = clf.best_params_['C']
    best_gamma = clf.best_params_['gamma']
    clf = svm.SVC(C=best_C, gamma=best_gamma)
    clf.fit(X_train, y_train)

    return clf, best_C, best_gamma


def svc_classifier_test(clf, X_test):
    y_pred = clf.predict(X_test)

    return y_pred


def logistic_classifier_train(X_train, y_train):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.35, random_state=42)
    logistic = linear_model.LogisticRegression()
    clf = GridSearchCV(logistic, param_grid={'C': np.logspace(-3, 3, 6)}, cv=cv, n_jobs=4, verbose=10)
    clf.fit(X_train, y_train)

    score_train = clf.score(X_train, y_train)

    print("Train_Score :{}".format(score_train))
