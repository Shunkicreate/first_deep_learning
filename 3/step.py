import matplotlib.pyplot as plt
import numpy as np

def step_function(x):
  return (x>0).astype(np.int)
  
x=np.arange(-5,5,0.01)
y=step_function(x)
plt.plot(x,y)
plt.show()