#!/usr/bin/env python
import numpy as np
import pdb 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
import math

# def get_gauss(u,p, sigma ):
# 	k = 1/(np.power(2*math.pi, 0.5))/(sigma**2)
# 	g = k*np.exp(-((p - u)**2)/(2*(sigma**2)))
# 	return g

def action_value(step_sz, eps, total_steps, runs, flag):
	N = 10
	#total_steps = 1000
	#runs = 2000
	Opt_a = np.zeros(total_steps)
	R_f = np.zeros(total_steps)

	for i in range(runs):

		step = 0;
		
		q_star = np.zeros(N) #np.random.uniform(0, 1, N)  #np.ones(N)
		#print("q_star: " , q_star)
		Q = np.zeros(N)	
		R = np.zeros(total_steps)
		noa = np.zeros(N)

		#eps = 0.1;
		#step_sz = 1.0/total_steps

		while step < total_steps:
			q_star = q_star + np.random.normal(0, 0.01, N)

			#pdb.set_trace()
			p = np.random.rand(1)
			if p[0] <= eps: #explore
				p2 = np.random.rand(1)[0]
				a = int(np.floor(p2*10))
				a = np.random.randint(0, 9)
				#R[step] = Q[a]
			else:
				a = np.argmax(Q)
				#R[step] = Q[a]

			noa[a] += 1;	
			if a == np.argmax(q_star):
				Opt_a[step] += 1;	

			R[step] = np.random.normal(q_star[a],1, 1)	
			if flag == 0:
				Q[a] = Q[a] + (1.0/noa[a])*(R[step] - Q[a])
			else:
				Q[a] = Q[a] + step_sz*(R[step] - Q[a])	

			step += 1;

		#addding all the rewards	
		R_f += R
		print("run completed = " , i)
		# print("Q: " , Q)
		# print("Q STAR" , q_star)

	R_f = R_f/runs
	Opt_a = Opt_a/runs


	return R_f, Opt_a

#INITIALIZING final reward and optimal action percentage
R = []
OA = []

total_steps = 10000
runs = 1000
step_sz = 1.0/total_steps
eps = 0.1

flag = 0;
R1, oa1 = action_value(step_sz, eps, total_steps, runs, flag)
R.append(R1)
OA.append(oa1)

#pdb.set_trace()

# total_steps = 1
# runs = 2
eps = 0.1
step_sz = 0.1
flag = 1;
R2, oa2 = action_value(step_sz, eps, total_steps, runs, flag)
R.append(R2)
OA.append(oa2)

X = np.arange(total_steps) #[i for i in range(t)]

C = ('green', 'blue')
label = ('sample average' , 'fixed step size')
#plt.plot(X, np.random(10, size=t))
line_label = []	
lines = []
for j in range(2):
	lines.append(plt.plot(X, R[j], color=C[j]))
	line_label.append(label[j])

plt.legend(line_label)
plt.ylabel('Average Reward' , size=10)
plt.xlabel('Steps', size=10)
plt.title("average sample")
# plt.ylim(0, 2)
plt.show()	

line_label = []	
lines = []
for j in range(2):
	lines.append(plt.plot(X, OA[j], color=C[j]))
	line_label.append(label[j])

plt.legend(line_label)
plt.ylabel('Optimal Action' , size=10)
plt.xlabel('Steps', size=10)
plt.ylim(0, 1)
plt.title("average sample")
plt.show()