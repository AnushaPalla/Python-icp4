from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
iris_datasets = datasets.load_iris()
x = iris_datasets.data
y = iris_datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = GaussianNB()
model.fit(x_train, y_train)
print(f"The Probability of training model is {model.score(x_train,y_train)}")
prediction = model.predict(x_test)
print("The Accuracy score using Naive bayes is : ", metrics.accuracy_score(y_test, prediction))