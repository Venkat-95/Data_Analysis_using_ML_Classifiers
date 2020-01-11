import pandas as pd
import numpy  as np
from sklearn import datasets, linear_model, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def preprocessing(file, imputer0, imputer1, scalar):
    data = pd.read_csv(file, na_values="?", header=None).values
    data0 = []
    data1 = []
    X0 = []
    X1 = []
    y0 = []
    y1 = []
    # print(type(data))
    for i in data:
        if (i[15] == 0):
            data0.append(i[0:16])
            X0.append(i[0:15])
            y0.append(i[15])
        else:
            data1.append(i[0:16])
            X1.append(i[0:15])
            y1.append(i[15])
    if (imputer0 == None and imputer1 == None):
        imputer0 = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
        imputer1 = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
        X0 = imputer0.fit_transform(X0)
        X1 = imputer1.fit_transform(X1)
    else:
        X0 = imputer0.transform(X0)
        X1 = imputer1.transform(X1)

    x = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)

    if scalar == None:
        scalar = StandardScaler()
        x = scalar.fit_transform(x)
    else:
        x = scalar.transform(x)

    return x, y, imputer0, imputer1, scalar


X_train, y_train, imputer0, imputer1, scalar = preprocessing("app.data", None, None, None)

C = np.logspace(-3, 3, 6)
gamma = np.logspace(-3, 3, 6)
parameters = {'C': C, 'gamma': gamma}
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.35, random_state=42)

T_clf = svm.SVC()
clf = GridSearchCV(estimator=T_clf, param_grid=parameters, n_jobs=4, cv=cv, verbose=10)
clf.fit(X_train, y_train)

mean = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))
best_C = clf.best_params_['C']
best_gamma = clf.best_params_['gamma']
clf = svm.SVC(C=best_C, gamma=best_gamma)
clf.fit(X_train, y_train)

X_test, y_test, imputer0, imputer1, scalar = preprocessing('app.test', imputer0, imputer1, scalar)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy = %.f%%" %(accuracy*100.0))


score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)

print("Training score: {}, Test score: {}".format(score_train, score_test))

data_unlabeled = pd.read_csv("app_unlabeled.test",na_values=["?"], header=None)
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imp.fit(X_train)
data_unlabeled=imp.transform(data_unlabeled)
data_unlabeled=scalar.transform(data_unlabeled)

predictions = clf.predict(data_unlabeled)
app_svm = np.column_stack((data_unlabeled,predictions))
np.savetxt("app_svm.test",app_svm)
