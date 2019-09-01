from sklearn import datasets
from sklearn import svm
from matplotlib import pyplot as plt  
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
boston =  datasets.load_boston()

lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1,
normalize=False)
predictions = lr.predict(boston.data)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))