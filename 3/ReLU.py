import matplotlib.pyplot as plt
import numpy as np

def relu_function(x):
  return np.maximum(0,x)

x=np.arange(-5,10,0.01)
y=relu_function(x)
plt.plot(x,y)
plt.show()