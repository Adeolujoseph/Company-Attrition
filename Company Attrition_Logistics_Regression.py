import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
data=pd.read_excel('CompanyAttrition.xlsx')
X=data.iloc[:, 0:9] #independednt variable
Y=data.iloc[:, 9] #dependent
data.info
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7,8])], remainder='passthrough')
X=ct.fit_transform(X)
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
Y=le.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state =0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print(model.score(X_test,y_test))
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
Accuracy= accuracy_score(y_test, y_pred)*100
Accuracy
prediction=pd.read_excel('CompanyAttrition.xlsx', sheet_name='Predictions')
P=prediction.iloc[:,0:9]
Py=prediction.iloc[:,9]
CT=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7,8])], remainder='passthrough')
P=CT.fit_transform(P)
Py=model.predict(P)
Py
PREDICTIONS=pd.DataFrame(prediction)
#PREDICTIONS.to_excel('CompanyAttritionResult.xlsx',sheet_name='pred', index=None, header=True)
#model.intercept_
#model.coef_

