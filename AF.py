import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+(np.e**(-x)))

x = np.arange(-6, 6, 0.1)
print(x)
plt.plot(x, sigmoid(x))
plt.title("sigmoid function")
plt.show()