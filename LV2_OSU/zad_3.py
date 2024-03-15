import numpy as np
import matplotlib . pyplot as plt

import numpy as np
import matplotlib . pyplot as plt

img = plt.imread ("road.jpg")
img = img [ :,:,0].copy()
plt.figure()
plt.imshow(img, cmap ="coolwarm")
plt.show()

img2 = img[::,200:400]
plt.imshow(img2, cmap="gray")
plt.show()

plt.imshow(np.rot90(img,3), cmap="gray")
plt.show()

plt.imshow(np.fliplr(img), cmap="gray")
plt.show()