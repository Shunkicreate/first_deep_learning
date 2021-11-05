import matplotlib.pyplot as plt
import numpy as np

def sigmoid_function(x):
  return 1/(1+np.exp(-x))

# x=np.arange(-5,5,0.01)
# y=sigmoid_function(x)
# plt.plot(x,y)
# plt.show()