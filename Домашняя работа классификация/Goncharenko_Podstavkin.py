import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

df = pd.read_csv("D:\\Project_with_Fedya\\6class.csv")
print(df.info())

#//////////////////////////LogisticRegression////////////////////////////////
kf = KFold(n_splits=5)
le = LabelEncoder()	#Создаём объект на основе LabelEncoder
for col in df.columns.values:	#Бежим в цикле по всем столбцам
    if df[col].dtypes=='object':	#Если тип нашего столбца равняется типу 'object'
        df[col] = le.fit_transform(df[col])	#Записываем в колонку те же данные, но с преобразованными категориальными признаками в числовые
X = df.drop('Star type',axis=1)	#Вектор признаков
y = df['Star type']	#Вектор ответов
scaler = MinMaxScaler()	#Создаёт объект, который позже будет нормализовывать данные 
X_norm = scaler.fit_transform(X)	#Передаём нормализованный Х
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=18)	#Делим данные на тстовые и тренировочные
logistic_regr = LogisticRegression()	#Создаём модель логистической регресии
logistic_regr.fit(X_train, y_train)	#Заполняем её данными
y_pred = logistic_regr.predict(X_test)	#Возвращает вектор ответов для X_test
print("--------------------------------LogisticRegression-----------------------------------")
print("Accuracy with MinMaxScaler: ", accuracy_score(y_test, y_pred))
print("Recall with MinMaxScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with MinMaxScaler: ", precision_score(y_test, y_pred, average = 'macro'))
cvs = cross_val_score(estimator = logistic_regr, X = X_train, y = y_train, cv = kf, scoring = 'accuracy')
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

scaler = StandardScaler()
X_st = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.3, random_state = 42)
logistic_regr = LogisticRegression()
logistic_regr.fit(X_train, y_train)
y_pred = logistic_regr.predict(X_test)
print("Accuracy with StandardScaler: ", accuracy_score(y_test, y_pred))
print("Recall with StandardScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with StandardScaler: ", precision_score(y_test, y_pred, average = 'macro'))
cvs = cross_val_score(estimator = logistic_regr, X = X_train, y = y_train, cv = kf, scoring = 'accuracy')
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Можно заметить, что стандартизированные данные работают лучше
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#//////////////////////////KNN////////////////////////////////
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_norm, y, test_size = 0.3, random_state = 42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cvs = cross_val_score(estimator = knn, X = X, y = y, cv = kf, scoring = 'accuracy')
print("--------------------------------KNN-----------------------------------")
print("Accuracy with MinMaxScaler: ", accuracy_score(y_test, y_pred))
print("Recall with MinMaxScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with MinMaxScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Accuracy with MinMaxScaler: ", accuracy_score(y_test, y_pred))
print("Recall with MinMaxScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with MinMaxScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Without cross val score: ", knn.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

scaler = StandardScaler()
X_st = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_st, y, test_size = 0.3, random_state = 42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cvs = cross_val_score(estimator = knn, X = X, y = y, cv = kf, scoring = 'accuracy')
print("Accuracy with StandardScaler: ", accuracy_score(y_test, y_pred))
print("Recall with StandardScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with StandardScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Accuracy with StandardScaler: ", accuracy_score(y_test, y_pred))
print("Recall with StandardScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with StandardScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Without cross val score: ", knn.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#И там и там всё работает отлично без cross val score
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#//////////////////////////Дерево решений////////////////////////////////
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("--------------------------------Дерево решений-----------------------------------")
print("Accuracy without StandardScaler and MinMaxScaler:", accuracy_score(y_test,y_pred_tree))
print("Recall without StandardScaler and MinMaxScaler:", recall_score(y_test,y_pred_tree, average = 'macro'))
print("Precision without StandardScaler and MinMaxScaler:", precision_score(y_test,y_pred_tree, average = 'macro'))
print(confusion_matrix(y_test, y_pred))

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_norm, y, test_size = 0.3, random_state = 42)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
cvs = cross_val_score(estimator = tree, X = X, y = y, cv = kf, scoring = 'accuracy')
print("Accuracy with MinMaxScaler: ", accuracy_score(y_test, y_pred))
print("Recall with MinMaxScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with MinMaxScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

scaler = StandardScaler()
X_st = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_st, y, test_size = 0.3, random_state = 42)
tree =  DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
cvs = cross_val_score(estimator = tree, X = X, y = y, cv = kf, scoring = 'accuracy')
print("Accuracy with StandardScaler: ", accuracy_score(y_test, y_pred))
print("Recall with StandardScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with StandardScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Результат сто процентный
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#//////////////////////////Метод опорных векторов////////////////////////////////
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_norm, y, test_size = 0.3, random_state = 42)
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("--------------------------------Метод опорных векторов-----------------------------------")
print("Accuracy with MinMaxScaler: ", accuracy_score(y_test, y_pred_svm)) 
print("Recall with MinMaxScaler: ", recall_score(y_test, y_pred_svm, average = 'macro'))  
print("Precision with MinMaxScaler: ", precision_score(y_test, y_pred_svm, average = 'macro'))
cvs = cross_val_score(estimator=svm, X=X, y=y, cv=kf, scoring='accuracy')
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

scaler = StandardScaler()
X_st = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_st, y, test_size = 0.3, random_state = 42)
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
cvs = cross_val_score(estimator = svm, X = X, y = y, cv = kf, scoring = 'accuracy')
print("Accuracy with StandardScaler: ", accuracy_score(y_test, y_pred))
print("Recall with StandardScaler: ", recall_score(y_test, y_pred, average = 'macro'))
print("Precision with StandardScaler: ", precision_score(y_test, y_pred, average = 'macro'))
print("Cross val score: ", cvs.mean())
print(confusion_matrix(y_test, y_pred))

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Результат работы с MinMaxScaller выше, чем с StandartScaller. Cross val score - вообще ужасный
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#