import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# Zad 1
unique_colors = np.unique(img_array_aprox, axis=0)
print("Broj razlicitih boja:", len(unique_colors))

# Zad 2, 3, 4
km = KMeans(n_clusters=5)
km.fit(img_array)

centroids=km.cluster_centers_
labels=km.predict(img_array)

img_array_approx = centroids[labels]
img_approx = np.reshape(img_array_approx, (w, h, d))

plt.figure()
plt.title("Promjenjena slika, K = 5")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()



km = KMeans(n_clusters=10)
km.fit(img_array)

centroids=km.cluster_centers_
labels=km.predict(img_array)

img_array_approx = centroids[labels]
img_approx = np.reshape(img_array_approx, (w, h, d))

plt.figure()
plt.title("Promjenjena slika, K = 10")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()
#Što je K manji slika vise odudara od originala


#5
# ucitaj sliku
img = Image.imread("imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# Zad 1
unique_colors = np.unique(img_array_aprox, axis=0)
print("Broj razlicitih boja:", len(unique_colors))

# Zad 2, 3, 4
km = KMeans(n_clusters=5, n_init="auto")
km.fit(img_array)

centroids=km.cluster_centers_
labels=km.predict(img_array)

img_array_approx = centroids[labels]
img_approx = np.reshape(img_array_approx, (w, h, d))

plt.figure()
plt.title("Promjenjena slika, K = 5")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()



km = KMeans(n_clusters=10, n_init="auto")
km.fit(img_array)

centroids=km.cluster_centers_
labels=km.predict(img_array)

img_array_approx = centroids[labels]
img_approx = np.reshape(img_array_approx, (w, h, d))

plt.figure()
plt.title("Promjenjena slika, K = 10")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()

#6
j = []
for n in range(1,10):
    km = KMeans(n_clusters=n, init="random", n_init=5)
    km.fit(img_array_aprox)
    colorLabels = km.predict(img_array_aprox)
    j.append(km.inertia_)

plt.plot(range(1,10),j)
plt.show()

#K = 2
#7
for i in range(0, n):
    boolArray = colorLabels == i
    boolArray = np.reshape(boolArray, (w,h))
    plt.figure()
    plt.title(f"binarna slika sa {i} grupa")
    plt.imshow(boolArray)
    plt.tight_layout()
    plt.show()