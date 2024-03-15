import numpy as np
import matplotlib . pyplot as plt

a = np.array ([1,2,3,3,1], np.float32 )
b = np.array ([1,2,2,1,1], np.float32 )

plt . plot (a , b , "b", linewidth =3 , marker =".", markersize = 15 )
plt . axis ([0.0,4.0,0.0,4.0])
plt . xlabel ("x os")
plt . ylabel ("y os")
plt . title ("slika1")
plt.show()