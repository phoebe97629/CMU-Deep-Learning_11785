import numpy as np

a = np.array([[1, 3, 5, 7],
             [6, 7, 9, 0]])

b = np.array([[1,0,1,0],
             [0,1,0,1]])


c = np.dot(a,b)

print(c)
