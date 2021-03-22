print("\n1. Check versions ..........")

# Python version

from matplotlib import pyplot
import matplotlib
import numpy
import scipy
import sys

from pandas import read_csv
import pandas
from pandas.plotting import scatter_matrix

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


print('Python: {}'.format(sys.version))
# scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))

print("\n2.1 Import libraries ...........")

print("\n2.2 Load dataset ...........")
url = "../Data/dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# print(dataset)

print("\n3.1 Summarize the dataset ...........")
print(dataset.shape)

print("\n3.2 Peek at the dataset ...........")
print(dataset.head(20))

print("\n3.3 Statistical summary of attributes ...........")
print(dataset.describe())

print("\n3.4 Class distribution (# of instances/rows that belong to each class)")
print(dataset.groupby('class').size())

# 4. Data Visualization
print("\n4.1 Univariate Plots (understand each attribute)")
print("\n4.1.1 Plot Box and Whisker ...........")
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

print("\n4.1.2 Plot histograms ...........")
dataset.hist()
pyplot.show()

print("\n4.2 Multivariate Plots (understand relationships between attributes)")
print("\n4.2 Plot Scatter Matrix ...........")
scatter_matrix(dataset)
pyplot.show()

# 5. Evaluate Some Algorithms
print("\n5.1 Create validation (Split-out) dataset")
array = dataset.values

X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("\n5.2 Test Harness...")
# Use stratified 10-fold cross-validation to estimate model accuracy.
# This will split our dataset into 10 parts, train on 9 and test on 1 
# and repeat for all combinations of train-test splits.
# Stratified means that each fold or split of the dataset will aim to have the
# same distribution of example by class as exist in the whole training dataset.

print("\n5.3 Build Models...")
# We don’t know which algorithms would be good on this problem 
# or what configurations to use.
# We get an idea from the plots that some of the classes are partially linearly 
# separable in some dimensions, so we are expecting generally good results.

# Let’s test 6 different algorithms:
# 1 Logistic Regression (LR)
# 2 Linear Discriminant Analysis (LDA)
# 3 K-Nearest Neighbors (KNN).
# 4 Classification and Regression Trees (CART).
# 5 Gaussian Naive Bayes (NB).
# 6 Support Vector Machines (SVM).
# This is good mixt of simple linear (LR,LDA),nonlinear (KNN,CART,NB,SVM) algorithms

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

print("\n5.4 Compare algorithms and select best model")
pyplot.boxplot(results, labels=names)
pyplot.title('5.4 Algorithm Comparison')
pyplot.show()
# SVM is the most accurate model with 98% score. Use this model as final model

print("\n6.1 Make Predictions")
# Fit the model on the entire training dataset and make predictions on the validation dataset.
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n6.2 Evaluate Predictions")
# Evaluate predictions by comparing them to the expected results in the validation set, 
# then calculate classification accuracy, as well as a confusion matrix 
# and a classification report.
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# See that the accuracy is 0.966 or about 96% on the hold out dataset.

print("\n------------ End Of File -------------")
