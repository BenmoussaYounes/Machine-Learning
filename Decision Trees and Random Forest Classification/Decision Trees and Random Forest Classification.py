from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # randomstate = 23 fix it to fix result
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

clf_r = RandomForestClassifier()
clf_r.fit(x_train, y_train)

print('DecisionTreeClassifier : ', clf.score(x_test, y_test))
print('RandomForestClassifier : ', clf_r.score(x_test, y_test))
