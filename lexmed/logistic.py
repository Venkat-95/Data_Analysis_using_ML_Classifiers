import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit


def preprocessing(file,imputer0,imputer1,scalar):

    data = pd.read_csv(file,na_values="?",header=None).values
    data0 = []
    data1 = []
    X0 = []
    y0 =[]
    X1 =[]
    y1=[]
    for i in data:
        if i[15]==0 :
            data0.append(i[0:16])
            X0.append(i[0:15])
            y0.append(i[15])
        else:
            data1.append(i[0:16])
            X1.append(i[0:15])
            y1.append(i[15])
    if(imputer0==None and imputer1 == None):
        imputer0 = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
        imputer1=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
        X0=imputer0.fit_transform(X0)
        X1=imputer1.fit_transform(X1)
    else:
        X0 = imputer0.transform(X0)
        X1 = imputer1.transform(X1)
    x = np.concatenate((X0,X1), axis=0)
    y = np.concatenate((y0,y1),axis=0)

    if(scalar==None):
        scalar= StandardScaler()
        x = scalar.fit_transform(x)
    else:
        x = scalar.transform(x)

    return x,y, imputer0,imputer1,scalar

X_train, y_train,imputer0,imputer1, scalar = preprocessing("app.data",None,None,None)

C = np.logspace(-3,3,6)
parameters = {'C': C}
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.35, random_state=42)
logistic = linear_model.LogisticRegression()
clf = GridSearchCV(logistic,parameters,cv=cv,n_jobs=4,verbose=10)
clf.fit(X_train,y_train)
X_test,y_test,imputer0,imputer1,scalar = preprocessing("app.test",imputer0,imputer1,scalar)
score_train = clf.score(X_train,y_train)
score_test = clf.score(X_test,y_test)
print("Train_Score :{}, Test_Score: {}".format(score_train,score_test))
data_unlabeled = pd.read_csv("app_unlabeled.test",na_values=["?"],header=None)
imp = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imp.fit(X_train)
data_unlabeled=imp.transform(data_unlabeled)
data_unlabeled=scalar.transform(data_unlabeled)

predictions = clf.predict(data_unlabeled)
app_logreg = np.column_stack((data_unlabeled,predictions))
np.savetxt("app_logreg.test",app_logreg)
