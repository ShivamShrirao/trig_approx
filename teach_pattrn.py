#!/usr/bin/env python3
import numpy as np
from time import time
from threading import Thread, RLock
from multiprocessing import Pool, Array, RLock

## Threads slow due to GIL in cpu intensive, shared but good for more I/O
## Processes pool works excellent. superfast. Resource in shared memory.
## Might be even faster without shared resource

inputs=np.array([[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
outputs=np.array([1,1,1,0,1,0,0,0])

# np.random.seed(1)
weights=(np.frombuffer(Array('d', 3*1).get_obj())).reshape(3,1)
weights+=2*np.random.random((3,1))-1
lk=RLock()

class neural_net:
	global weights
	w8s=weights
	def __init__(self):
		self.bias=4	#np.random.randint(10)

	def _sigmoid(self, x):
		return 1/(1+np.exp(-x))
		
	def think(self, inputs):
		return (self._sigmoid(np.dot(inputs, self.w8s).T-self.bias))

	def train(self, inputs, outputs, num):
		global lk
		for i in range(num):
			out = self.think(inputs)
			err = outputs-out
			adj = np.dot(inputs.T, (err*out*(1-out)).T)
			lk.acquire()
			self.w8s+=adj
			if not i%1000:
				print('\rProgress: ',i*100/num,' %',end='')
			lk.release()

nn=neural_net()
iterations=100000
thds=4
t=time()
if 0:
	threads=[]
	iterations=(iterations//thds)
	for i in range(thds):
		threads.append(Thread(target=nn.train, args=(inputs, outputs, iterations)))
		threads[-1].start()

	for thrd in threads:
		thrd.join()
else:
	# nn.train(inputs, outputs, iterations)
	iterations=(iterations//thds)
	Pool().starmap(nn.train, [(inputs, outputs, iterations) for i in range(thds)])

print()
print("Time:",(time()-t))
print(weights)
print('x:',iterations*thds)
print(nn.think([0,1,1]))
print(nn.think([1,0,1]))
print(nn.think([1,0,0]))
print(nn.think([1,1,1]))
print(nn.think([0,0,1]))
print(nn.think([0,0,0]))