#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import time

sd=np.random.randint(100)
sd=34
np.random.seed(sd)
rg,nrons=np.random.randint(10,20),np.random.randint(10,40)
X = np.array([[np.random.uniform(0,np.pi)] for i in range(rg)])
y = np.sin(X)

class neural_net:
	"""docstring for neural_net"""
	def __init__(self, X, y):
		self.X	= X
		self.y	= y
		self.w1	= 2*np.random.rand(self.X.shape[1],nrons)-1
		self.b1	= 2*np.random.rand(self.X.shape[1],nrons)-1
		self.w2	= 2*np.random.rand(self.X.shape[1],nrons)-1
		self.b2	= np.random.rand()-1
	
	def sigmoid(self,x):
		return 1.0/(1+ np.exp(-x))

	def sigmoid_der(self,x):
		return x * (1.0 - x)

	def tanh_der(self, x):
		return (1-x**2)

	def think(self, X):
		self.X	= X
		z 		= (np.dot(self.X,self.w1)+self.b1)
		self.a 	= self.sigmoid(z)
		z1 		= (np.dot(self.a,self.w2.T)+self.b2)
		out		= self.sigmoid(z1)
		return out

	def train(self,X,y, iterations):
		self.X, self.y = X, y
		for i in range(iterations):
			self.out= self.think(self.X)
			d_c_z1	= (2*(self.y-self.out)*self.sigmoid_der(self.out))
			d_c_w2	= np.dot(d_c_z1.T,self.a)
			d_c_b2	= d_c_z1.sum()
			d_c_z	= (np.dot(d_c_z1,self.w2)*self.sigmoid_der(self.a))
			d_c_w1	= np.dot(self.X.T, d_c_z)
			d_c_b1	= d_c_z.sum()
			self.w1+=d_c_w1
			self.b1+=d_c_b1
			self.w2+=d_c_w2
			self.b2+=d_c_b2
			# if not i%100:
				# print('\rProgress: ',i*100/iterations,' %',end='')
		print()
		self.cost=((self.y-self.out)**2).sum()

nn = neural_net(X,y)
x = np.arange(0,np.pi,0.01)
x = x.reshape(x.shape[0],1)
plt.plot(x*180/np.pi, nn.think(x), color= "yellow") 
t=time()
nn.train(X,y,100000)
print("Time:",(time()-t))
print(sd,rg,nrons,"Cost:",nn.cost)
print(nn.think(0))
print(nn.think(np.pi/6))
print(nn.think(np.pi/2))
y = np.sin(x)
plt.plot(x*180/np.pi, y, color= "green")
plt.plot(x*180/np.pi, nn.think(x), color= "red")
plt.scatter(sd, nn.think(sd), color= "blue", marker= "*") 
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('wave')
plt.show()
print(sd,rg,nrons,"Cost:",nn.cost)