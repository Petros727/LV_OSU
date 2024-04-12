import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# 1
# A)
KNN_train = KNeighborsClassifier (n_neighbors =5)
KNN_train.fit(X_train_n, y_train)
KNN_test = KNN_train.predict(X_test_n)
print(f"Tocnost podataka KNN-a: {accuracy_score(y_test, KNN_test)}")
plot_decision_regions(X_train_n, y_train, classifier=KNN_train)
plt.show()
# B)
KNN_K1 = KNeighborsClassifier (n_neighbors =1)
KNN_K1.fit(X_train_n, y_train)
KNN_K100 = KNeighborsClassifier (n_neighbors =100)
KNN_K100.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=KNN_K1)
plt.title("K=1")
plt.show()
plot_decision_regions(X_train_n, y_train, classifier=KNN_K100)
plt.title("K=100")
plt.show()

# 2
KNN_model = KNeighborsClassifier()
param_grid = {'KNN__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}
pipe = Pipeline(steps=[("sc", sc), ("KNN", KNN_model)])
knn_gscv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_gscv.fit(X_train, y_train)
print(knn_gscv.best_params_)
print(knn_gscv.best_score_)

# 3
SVM_model = svm.SVC(kernel="rbf")
SVM_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.title("Kernel: RBF, C=default, gamma=default")
plt.show()
# C = 0.1
SVM_C1_model = svm.SVC(kernel="rbf", C=0.1)
SVM_C1_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_C1_model)
plt.title("Kernel: RBF, C=0.1, gamma=default")
plt.show()
# C = 10
SVM_C2_model = svm.SVC(kernel="rbf", C=10)
SVM_C2_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_C2_model)
plt.title("Kernel: RBF, C=10, gamma=default")
plt.show()
# gamma = 0.1
SVM_gamma1_model = svm.SVC(kernel="rbf", gamma=0.1)
SVM_gamma1_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_gamma1_model)
plt.title("Kernel: RBF, C=default, gamma=0.1")
plt.show()
# gamma = 10
SVM_gamma2_model = svm.SVC(kernel="rbf", gamma=10)
SVM_gamma2_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_gamma2_model)
plt.title("Kernel: RBF, C=default, gamma=10")
plt.show()
# kernel: linear
SVM_linear_model = svm.SVC(kernel="linear")
SVM_linear_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_linear_model)
plt.title("Kernel: linear, C=default, gamma=default")
plt.show()
# kernel: poly
SVM_poly_model = svm.SVC(kernel="poly")
SVM_poly_model.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=SVM_poly_model)
plt.title("Kernel: poly, C=default, gamma=default")
plt.show()

# 4
SVM_model = svm.SVC()
param_grid = {'C': np.arange(1, 10), 'gamma': np.arange(0.0, 5.0)}
SVM_gs = GridSearchCV(SVM_model, param_grid, cv=5)
SVM_gs.fit(X_train_n, y_train)
print(SVM_gs.best_params_, SVM_gs.best_score_)
print (SVM_gs.cv_results_)


