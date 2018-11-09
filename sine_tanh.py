#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import time

sd=np.random.randint(100)
try:
	sd=int(argv[1])
except:
	# sd=73
	sd=19#53#75#124
	pass

qet=1
np.random.seed(sd)
rg,nrons=np.random.randint(10,40),np.random.randint(15,50)
# rg,nrons=39,44
X = np.array([[np.random.uniform(0,np.pi*2)] for i in range(100)])
# X[1]=np.pi/6
# y = np.sin(X)
y = np.sin(X)

class neural_net:
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
		z 		= (np.dot(X,self.w1)+self.b1)
		z=z/100
		self.a 	= np.tanh(z)
		z1 		= (np.dot(self.a,self.w2.T)+self.b2)
		z1=z1/100
		out		= np.tanh(z1)
		return out

	def train(self,X,y, iterations):
		self.X, self.y = X, y
		for i in range(iterations):
			self.out= self.think(self.X)
			d_c_z1	= (2*(self.y-self.out)*self.tanh_der(self.out))
			d_c_w2	= (np.dot(d_c_z1.T,self.a))/100
			d_c_b2	= (d_c_z1.sum())/100
			d_c_z	= (np.dot(d_c_z1,self.w2)*self.tanh_der(self.a))
			d_c_w1	= (np.dot(self.X.T, d_c_z))/100
			d_c_b1	= (d_c_z.sum())/100
			self.w1+=d_c_w1
			self.b1+=d_c_b1
			self.w2+=d_c_w2
			self.b2+=d_c_b2
			if (i>1400) and qet:
				if (not i%1000):
					plt.plot(x*180/np.pi, nn.think(x), color= "pink")
					print('\rProgress:',i*100/iterations,' %',end='')
			elif qet:
				if (not i%10) and (i<100):
					plt.plot(x*180/np.pi, nn.think(x), color= "yellow")
				elif (not i%50) and (i<500):
					plt.plot(x*180/np.pi, nn.think(x), color= "yellow")
				elif (not i%100) and (i<1400):
					plt.plot(x*180/np.pi, nn.think(x), color= "yellow")
		print()

nn = neural_net(X,y)
x = np.arange(0,np.pi*2,0.01)
x = x.reshape(x.shape[0],1)
# y2 = np.sin(x)
y2 = np.sin(x)
if qet:
	plt.plot(x*180/np.pi, nn.think(x), color= "yellow")

t=time()
nn.train(X,y,100000)
nn.train(X,y,100000)
nn.train(X,y,100000)
nn.train(X,y,100000)
nn.train(X,y,100000)

print("Time:",(time()-t))

if qet:
	plt.plot(x*180/np.pi, y2, color= "green")
	plt.plot(x*180/np.pi, nn.think(x), color= "red")
nn.cost=((y2-nn.think(x))**2).mean()*10
print("Cost:",nn.cost)

if qet:
	plt.scatter(sd, nn.think(sd), color= "blue", marker= "*") 
	plt.xlabel('x - axis')
	plt.ylabel('y - axis')
	plt.title('wave')
	plt.show()
print(sd,rg,nrons,"Cost:",nn.cost)