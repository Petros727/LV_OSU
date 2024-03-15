import numpy as np
import matplotlib . pyplot as plt

data = np.loadtxt("data (1).csv" , delimiter="," , skiprows= 1)
print("Istrazivanje je provedeno na:", data.shape[0])
height = data[:,1]
weight = data[:,2]

plt.scatter(height,weight)
plt.xlabel("visina")
plt.ylabel("debljina")
plt.title("odnos visine i debljine")
plt.show()


height_every_50 = data[:,1][::50]
weight_every_50 = data[:,2][::50]

plt.scatter(height_every_50,weight_every_50)
plt.xlabel("visina")
plt.ylabel("debljina")
plt.title("odnos visine i debljine svake pedesete osobe")
plt.show()

print("maksimalna visina je:",max(height))
print("minimalna visina je:",min(height))
print("prosjecna visina je:" , height.mean())

male = (data[:,0] == 1)
female = (data[:,0] == 0)


print("maksimalna visina muskarca je:",data[male,1].max())
print("minimalna  visina muskarca je:",data[male,1].min())
print("prosjecna visina muskarca je:" , data[male,1].mean())

print("maksimalna visina zene je:",data[female,1].max())
print("minimalna visina zene je:",data[female,1].min())
print("prosjecna visina zene je:" , data[female,1].mean())