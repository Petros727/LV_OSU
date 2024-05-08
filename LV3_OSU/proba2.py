import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Učitavanje Iris dataset-a
iris = load_iris()

# a) Scatter dijagram odnosa duljine latice i širine čašice za Virginicu (zelenom bojom) i Setosu (sivom bojom)
virginica = iris.data[iris.target == 2]  # Podaci samo za Virginicu
setosa = iris.data[iris.target == 0]  # Podaci samo za Setosu

plt.scatter(virginica[:, 0], virginica[:, 1], c='green', label='Virginica')
plt.scatter(setosa[:, 0], setosa[:, 1], c='grey', label='Setosa')

plt.xlabel('Duljina latice')
plt.ylabel('Širina čašice')
plt.title('Odnos duljine latice i širine čašice za Virginicu i Setosu')
plt.legend()

plt.show()

max_sepal_width_per_class = [max(iris.data[iris.target == i][:, 1]) for i in range(3)]

plt.bar(iris.target_names, max_sepal_width_per_class, color=['blue', 'orange', 'green'])
plt.xlabel('Klasa cvijeta')
plt.ylabel('Najveća vrijednost širine čašice')
plt.title('Najveća vrijednost širine čašice za sve tri klase cvijeta')
plt.show()

# c) Izračun broja jedinki klase Setosa s većom širinom čašice od prosječne širine čašice te klase
setosa_width = iris.data[iris.target == 0][:, 1]  # Širina čašice samo za Setosu
mean_setosa_width = setosa_width.mean()  # Prosječna širina čašice za Setosu
num_greater_than_mean = sum(setosa_width > mean_setosa_width)  # Broj jedinki s većom širinom čašice od prosjeka

print("Broj jedinki klase Setosa s većom širinom čašice od prosječne širine čašice te klase:", num_greater_than_mean)

#2

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

# Učitavanje Iris dataset-a
iris = load_iris()
X = iris.data  # Značajke
y_true = iris.target  # Stvarne oznake klasa

# a) Pronalaženje optimalnog broja klastera K pomoću metode lakta
distortions = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# b) Grafički prikaz metode lakta
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('Broj klastera K')
plt.ylabel('Izobličenje')
plt.title('Metoda lakta za pronalaženje optimalnog broja klastera K')
plt.show()

# Odabir optimalnog broja klastera K (npr. na temelju vizualne inspekcije metode lakta)
optimal_K = 3

# c) Primjena algoritma K-srednjih vrijednosti
kmeans = KMeans(n_clusters=optimal_K)
kmeans.fit(X)
y_pred = kmeans.labels_  # Predviđene oznake klastera

# d) Prikaži dobivene klastere pomoću dijagrama raspršenja
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Klasteri')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=200, label='Centroidi')
plt.xlabel('Duljina latice')
plt.ylabel('Širina latice')
plt.title('K-srednje vrijednosti clustering rezultat')
plt.legend()
plt.show()

# e) Usporedi dobivene klase s stvarnim vrijednostima i izračunaj točnost klasifikacije
accuracy = accuracy_score(y_true, y_pred)
print("Točnost klasifikacije:", accuracy)

