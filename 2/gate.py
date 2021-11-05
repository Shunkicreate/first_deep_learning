import numpy as np
def pre_AND(x1,x2):
  w1,w2,theta=0.5,0.5,0.7
  tmp=x1*w1+x2*w2
  if tmp<=theta:
    return 0
  elif tmp >theta:
    return 1
  
def AND(x1,x2):
  x=np.array([x1,x2])
  w=np.array([0.5,0.5])
  b=-0.7
  tmp=np.sum(w*x)+b
  if(tmp<=0):
    return 0
  else:
    return 1
  
def NAND(x1,x2):
  x=np.array([x1,x2])
  w=np.array([0.5,0.5])
  b=-0.7
  tmp=np.sum(w*x)+b
  if(tmp>=0):
    return 0
  else:
    return 1
  
def OR(x1,x2):
  x=np.array([x1,x2])
  w=np.array([0.5,0.5])
  b=-0.2
  tmp=np.sum(w*x)+b
  if(tmp<=0):
    return 0
  else:
    return 1

def XOR(x1,x2):
  x=NAND(x1,x2)
  y=OR(x1,x2)
  return AND(x,y)

  
  
  
print(AND(0,0),AND(0,1),AND(1,0),AND(1,1))
print(NAND(0,0),NAND(0,1),NAND(1,0),NAND(1,1))
print(OR(0,0),OR(0,1),OR(1,0),OR(1,1))
print(XOR(0,0),XOR(0,1),XOR(1,0),XOR(1,1))
    
  