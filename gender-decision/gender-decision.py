from sklearn import tree, svm, neighbors, linear_model
from sklearn.metrics import accuracy_score
import numpy as np

clf = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = neighbors.KNeighborsClassifier()
clf4 = linear_model.LogisticRegression()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
prediction1 = clf.predict(X)
acc_clf = accuracy_score(Y, prediction1)
print('Accuracy of Decision Tree is: ', acc_clf)

n = [[172,76,43],]
output = clf.predict(n)
print('172,76,43 is a : ',output)

clf2 = clf2.fit(X, Y)
prediction2 = clf2.predict(X)
acc_clf2 = accuracy_score(Y, prediction2)
print('Accuracy of Support Vector Machine is: ', acc_clf2)

clf3 = clf3.fit(X, Y)
prediction3 = clf3.predict(X)
acc_clf3 = accuracy_score(Y, prediction3)
print('Accuracy of K-Nearest Neighbors is: ', acc_clf3)

clf4 = clf4.fit(X, Y)
prediction4 = clf4.predict(X)
acc_clf4 = accuracy_score(Y, prediction4)
print('Accuracy of Logistic Regression is: ', acc_clf4)

# CHALLENGE compare their reusults and print the best one!
max_idx = np.argmax([clf, clf2, clf3, clf4])
max_classifier = {0: 'Tree', 1: 'SVM', 2: 'KNN', 3: 'LR'}
print('The best classifier is: ', max_classifier[max_idx])
