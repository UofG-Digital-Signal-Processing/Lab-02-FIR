import numpy as np


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tmp = np.copy(a[0:int(len(a) / 2)])
a[0:int(len(a) / 2)] = a[int(len(a) / 2):len(a)]
a[int(len(a) / 2):len(a)] = tmp
print(a)