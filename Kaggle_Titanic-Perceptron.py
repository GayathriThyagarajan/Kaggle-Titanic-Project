import pandas as pd
import numpy as np
from scipy.stats import randint
from scipy.stats import uniform

# Reading the data
d = pd.read_csv("train.csv")
# Collect the X-values into X-matrix
X = d[['Pclass','Sex','Age','Fare']]
# Collect y-values
y = d.Survived

# Converting categorical values into numbers
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
X['Sex'] = class_le.fit_transform(X['Sex'])

# Impute missing values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean',axis=0)
imr = imr.fit(X)
X = imr.transform(X)

# Splitting data  
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)

 
d_test = pd.read_csv("test.csv")

X_test = d_test[['Pclass','Sex','Age','Fare']]
X_test['Sex'] = class_le.fit_transform(X_test['Sex'])
imr = imr.fit(X_test)
X_test = imr.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_val_std = sc.transform(X_val)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_val_std)
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))

sc.fit(X_test)
X_test_std = sc.transform(X_test)
y_test = ppn.predict(X_test_std)
y_test = pd.DataFrame(y_test)

d1 = d_test['PassengerId']
d2 = y_test
submission = pd.concat([d1, d2], axis=1)
submission.columns.values[1] = "Survived"
submission.to_csv("submission.csv",index=False)