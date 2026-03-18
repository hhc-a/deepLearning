#import
import numpy as np
# create ndarray
arr1d = np.array([1,2,3,4])
print(arr1d)
print(type(arr1d))

arr2d = np.array([[1,2,3,4],
                 [5,6,7,8]])
print(arr2d)
print(type(arr2d))

#
#np.arange
arr2d02 = np.ones((2,4)) # "ones can change to" empty, zeros(...), ones(...), arange(100)
print(arr2d02)
print(type(arr2d02))
print(arr2d02.ndim)
print(arr2d02.shape)

# math: + - * / ...
print("Math op...")
arr2d_result = arr2d + arr2d02
print(arr2d_result)
print(arr2d - arr2d02)
print(arr2d * arr2d02)
print(arr2d / arr2d02)
print(arr2d > arr2d02)
print(arr2d == arr2d02)
print(arr2d < arr2d02)

# matrix operation
arr2d03 = np.array([1,0,1,0]).T #4*1 T=tranform轉向
print( arr2d.dot(arr2d03))
print( arr2d @ arr2d03)
print( np.outer(arr2d, arr2d03))

# statical ...
print( arr2d.sum())
print( arr2d.sum(axis=1))
print( arr2d.max(axis=0))
print( arr2d.mean())
print( arr2d.std())

# shape
print(arr2d.ravel()) #攤平
print(arr2d.reshape((4,2)))

#range
for i in range(0, 10, 2):
    print(i, end=',')
print()
# indexing, slice
print(arr2d[1,2]) #7
print(arr2d[0:2,2]) #3 7
print(arr2d[0:2:2,2]) #3
print(arr2d[0:2:1,1:4:2])