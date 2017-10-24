from scipy.spatial import distance
#to find euc dst
def euc(a, b):
        return distance.euclidean(a,b)  

class myKNN():
	def fit(self,xtrain,ytrain):
		self.xtrain = xtrain
		self.ytrain = ytrain
	
	def predict(self, xtest):
		pred = []
		for row in xtest:
			lab = self.closest(row)
			pred.append(lab)
		return pred
#closest distance	
	def closest(self, row):
		bestdst = euc(row, self.xtrain[0])
		bestin = 0
		for i in range(1, len(self.xtrain)):
			dst = euc(row, self.xtrain[i])
			if dst < bestdst:
				bestdst = dst
				bestin = i
		return self.ytrain[bestin]
from sklearn import neighbors
from sklearn.datasets import load_iris
iris = load_iris()

feat = iris.data
lab = iris.target
#split data
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feat,lab,test_size=.3)

clf = myKNN()
clf.fit(xtrain, ytrain)
 
p = clf.predict(xtest)

#accuracy
from sklearn.metrics import accuracy_score

print("Accuracy = ",accuracy_score(ytest, p))

