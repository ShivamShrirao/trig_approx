#!/usr/bin/env python3
import numpy as np
from time import time
from multiprocessing import RLock, Manager, Array, Process
from threading import Thread, RLock

## Yeah this is slower for shared resource cpu. But has good implementations if
## performed over network multiple and without shared resource
## Todo pipes and queues

if __name__ == '__main__':
	inputs=np.array([[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
	outputs=np.array([1,1,1,0,1,0,0,0])

	# np.random.seed(1)
	m=Manager()
	# weights=(np.frombuffer(Array('d', 3*1).get_obj())).reshape(3,1)
	# weights+=2*np.random.random((3,1))-1
	weights=m.list()
	weights.append((2*np.random.random((3,1))-1))

class neural_net:
	global weights
	def _sigmoid(self, x):
		return 1/(1+np.exp(-x))
		
	def think(self, inputs):
		return self._sigmoid(np.dot(inputs, weights[0]).T-4)

	def train(self, inputs, outputs, num):
		global lk
		for i in range(num):
			out = self.think(inputs)
			err = outputs-out
			adj = np.dot(inputs.T, (err*out*(1-out)).T)
			lk.acquire()
			weights[0]+=adj
			lk.release()
			if not i%1000:
				print('\rProgress: ',i*100/num,' %',end='')
		# print(weights[0])

if __name__ == '__main__':
	nn=neural_net()
	iterations=100000
	t=time()
	thds=4
	lk=RLock()
	# nn.train(inputs, outputs, iterations)
	iterations=(iterations//thds)
	p=Process(target=nn.train, args=(inputs, outputs, iterations))
	p.start()
	p.join()

	print()
	print("Time:",(time()-t))
	print(weights[0])
	print('x:',iterations*thds)
	print(nn.think([0,1,1]))
	print(nn.think([1,0,1]))
	print(nn.think([1,0,0]))
	print(nn.think([1,1,1]))
	print(nn.think([0,0,1]))
	print(nn.think([0,0,0]))