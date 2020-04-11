import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

def preprocessing(file, imputer0, imputer1, scalar):
#error_bad_lines=False
    data = pd.read_csv(file, na_values="?", header=None).values
    data0 = []
    data1 = []
    X0 = []
    X1 = []
    y0 = []
    y1 = []
    # print(type(data))
    for i in data:
        if (i[200] == 0):
            data0.append(i[0:200])
            X0.append(i[0:200])
            y0.append(i[200])
        elif (i[200]==1):
            data1.append(i[0:200])
            X1.append(i[0:200])
            y1.append(i[200])
        else:
            continue
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


X_train, y_train, imputer0, imputer1, scalar = preprocessing("data2.csv", None, None, None)

pca = PCA(.95)
X_train = pca.fit_transform(X_train)
app_XTrain = np.column_stack(X_train)
np.savetxt("app_XTrain.train", app_XTrain)
C = np.logspace(-3, 3, 6)
gamma = [0.001,0.01,0.1,1,100,1000]
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

X_test, y_test, imputer0, imputer1, scalar = preprocessing('data1.csv', imputer0, imputer1, scalar)

X_test=pca.transform(X_test)

app_XTest = np.column_stack(X_test)
np.savetxt("app_XTest.test", app_XTest)
y_pred = clf.predict(X_test)
print(y_test)
print(y_pred)
accuracy = accuracy_score(y_pred,y_test,)
print("Accuracy = %.f%%" %(accuracy*100.0))
