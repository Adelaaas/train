import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Нормализованные данные
df = pd.read_csv("D:\\DataSet\\diabetes.csv")
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_norm, y, test_size=1/3, random_state=42)
scores_norm = []
for i in range(1, 15):
	tree = DecisionTreeClassifier(max_depth=i, random_state=42)
	tree.fit(X_train, y_train)
	y_pred_tree = tree.predict(X_test)
	scores_norm.append(tree.score(X_test, y_test))
print(scores_norm)
max_score = max(scores_norm)
scores_ind_norm = [i for i, v in enumerate(scores_norm) if v == max_score]
print(scores_ind_norm)
#[0.71875, 0.71875, 0.69140625, 0.72265625, 0.7578125, 0.71484375, 0.7421875, 0.7265625, 0.72265625, 0.703125, 0.703125, 0.703125, 0.703125, 0.703125]
#[4]

#Стандартизированные данные
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']
scaler = StandardScaler()
X_st = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.3, random_state = 42)
scores_stand = []
for i in range(1, 15):
  treee = DecisionTreeClassifier(max_depth=i, random_state=42)
  tree.fit(X_train, y_train)
  aswer = tree.predict(X_test)
  scores_stand.append(tree.score(X_test, y_test))
print(scores_stand)
max_score = max(scores_stand)
scores_ind_st = [i for i, v in enumerate(scores_stand) if v == max_score]
print(scores_ind_st)
#[0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013, 0.7012987012987013]
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Можно заметить, что нормализованные данные работают лучше
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#Логиситческая регрессия с помощью кросс-валидации
kf = KFold(n_splits=5)
log = LogisticRegression()
cvs = cross_val_score(estimator=log, X=X, y=y, cv=kf, scoring='accuracy')
print("1: ",cvs)
print("1: ",cvs.mean())
#[0.77272727 0.72077922 0.75324675 0.82352941 0.77777778]
#0.7696120872591461

#Логиситческая регрессия без помощи кросс-валидации
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_old = LogisticRegression()
log_old.fit(X_train, y_train)
answers_pred = log_old.predict(X_test)
print("1_old: ",log_old.score(X_test, y_test))
#0.7402597402597403

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Можно заметить, что логиситческая регрессия с помощью кросс-валидации работает лучше, чем без неё
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#Метод ближайших соседей с помощью кросс-валидации
kf = KFold(n_splits=5)
knn = KNeighborsClassifier()
cvs = cross_val_score(estimator=knn, X=X, y=y, cv=kf, scoring='accuracy')
print("2: ",cvs)
print("2: ",cvs.mean())
#[0.72727273 0.69480519 0.73376623 0.76470588 0.69934641]
#0.723979288685171

#Метод ближайших соседей без помощи кросс-валидации
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn_old = KNeighborsClassifier()
knn_old.fit(X_train,y_train)
answers = knn_old.predict(X_test)
print("2_old: ",knn_old.score(X_test,y_test))
#0.6883116883116883

#							^							 #
#							|							 #
#							|							 #
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#Можно заметить, что метод ближайших соседей с помощью кросс-валидации работает лучше, чем без неё
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#