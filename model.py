from sklearn.tree import DecisionTreeClassifier
features = [[140,0],[130,0],[150,1],[170,1]]
#Bumby -0 Smooth- 1
#apple-1 orange-0
labels = [0,0,1,1]

clf = DecisionTreeClassifier()
#Train
clf.fit(features,labels)
p = clf.predict([[160,1]])
print("Prediction= ",p)
