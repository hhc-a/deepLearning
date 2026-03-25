import numpy as np

def one_hot_encoding(raw, num):
    result = []
    for ele in raw:
        arr = np.zeros(num)
        np.put(arr, ele, 1)
        result.append(arr)
        
    return np.array(result)
    
digits = np.array([1, 8, 5, 4])

one_hot = one_hot_encoding(digits, 10)
print(digits)
print(one_hot)