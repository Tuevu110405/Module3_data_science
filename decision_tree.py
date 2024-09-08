#classification tasks

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#Load data sets
iris_X, iris_y = datasets.load_iris(return_X_y=True)

#Split train : test = 8:2
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=42)

#Define model
dt_classifier = DecisionTreeClassifier()

#Train model
dt_classifier.fit(X_train, y_train)

#Predict and evaluate
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# regression tasks

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor 

#Load data sets
machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_target = machine_cpu.target


#Split train : test = 8:2
X_train, X_test, y_train, y_test = train_test_split(machine_cpu.data, machine_cpu.target, test_size=0.2, random_state=42)

#Define model
dt_regressor = DecisionTreeRegressor()

#Train model
dt_regressor.fit(X_train, y_train)