from sklearn import neighbors
from sklearn.datasets import load_iris
iris = load_iris()

feat = iris.data
lab = iris.target
#split data
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feat,lab,test_size=.3)

clf = neighbors.KNeighborsClassifier()
clf.fit(xtrain, ytrain)
 
p = clf.predict(xtest)

#accuracy
from sklearn.metrics import accuracy_score

print("Accuracy = ",accuracy_score(ytest, p))

