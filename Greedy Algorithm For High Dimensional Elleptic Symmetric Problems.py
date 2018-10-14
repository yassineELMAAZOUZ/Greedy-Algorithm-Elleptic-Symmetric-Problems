# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 17:27:30 2017
@author: yassine

"""
#######################################################################################################
#import the necessary modules.
from scipy.integrate import quad
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################################################################################################
#define the constants
I=20
x_vec = np.linspace(0,1,I+1)
m=100   # Nombre d'itérations (du point fixe)
n=20    # Nombre de composantes de la solution finale
h=1./I  # Taille du maillage
#######################################################################################################

# f(x,y) = (f_1_1(x),f_2_1(y)) + (f_1_2(x),f_2_2(y)) (le second membre)

def f_1_1(x):
    return np.cos(3*np.pi*x)
def f_2_1(y):
    return np.cos(3*np.pi*y)

def f_1_2(x):
    return x**2
def f_2_2(y):
    return y*np.exp(np.sin(y))


def phi(i):     # returns the functions phi_i
    return (lambda x: np.maximum(0,1-np.abs(x-x_vec[i])/h ))
    
def times(f,g): # returns the products of f and g.
    return (lambda x: f(x)*g(x))

#######################################################################################################
#Storing the necessary matrices and vectors.

F_1_1 = np.array([quad(times(f_1_1,phi(i)),0,1)[0] for i in range(0,I+1)])
F_2_1 = np.array([quad(times(f_2_1,phi(i)),0,1)[0] for i in range(0,I+1)])    

F_1_2 = np.array([quad(times(f_1_2,phi(i)),0,1)[0] for i in range(0,I+1)])
F_2_2 = np.array([quad(times(f_2_2,phi(i)),0,1)[0] for i in range(0,I+1)])    


M = sparse.diags([h/6, 2*h/3, h/6], [-1, 0, 1], shape=(I+1, I+1),format = 'csc')
M[0,0]/=2
M[I,I]/=2

D = sparse.diags([-1./h,2./h,-1./h], [-1,0,1] , shape = (I+1,I+1),format = 'csc')
D[0,0] /=2
D[I,I] /=2

###############################################################################


U=[] # The list in which the vectors R_k and S_k are stored.


#Defining the functions used in the equation system in question 8.
def M_function(V):
    return np.dot(V.transpose(),(D+M)*V)*M +  np.dot(V.transpose(),M*V)*D


def F_function(V):
    Result = np.dot(V.transpose(),F_2_1)*F_1_1 + np.dot(V.transpose(),F_2_2)*F_1_2
    if len(U)==0: return Result        
    for L in U:
        Result -= np.dot(V.transpose(),D*L[1])*(M*L[0])
        Result -= np.dot(V.transpose(),M*L[1])*(D*L[0])
        Result -= np.dot(V.transpose(),M*L[1])*(M*L[0])
    return Result

def G_function(V):
    Result = np.dot(V.transpose(),F_1_1)*F_2_1 + np.dot(V.transpose(),F_1_2)*F_2_2
    if len(U)==0:
        return Result
    for L in U:
        Result -= np.dot(V.transpose(),D*L[0])*(M*L[1])
        Result -= np.dot(V.transpose(),M*L[0])*(D*L[1])
        Result -= np.dot(V.transpose(),M*L[0])*(M*L[1])
    return Result

    
def iteration(m):# the procedure that computes the vectors (R_k, S_k) and fill the list U.
    S= np.random.normal(10,4,I+1)
    for i in range(m):
        MS=M_function(S)
        lu_MS =  linalg.splu(MS, permc_spec = 'NATURAL')
        R= lu_MS.solve(F_function(S))
        MR=M_function(R)
        lu_MR =  linalg.splu(MR, permc_spec = 'NATURAL')
        S= lu_MR.solve(G_function(R))
        
    U.append([R,S])

#Filling the list U.
for i in range(n):
    iteration(m)


Y,X = np.meshgrid(x_vec,x_vec)

#Constructing the solution u_n from the vectors r_k,s_k
U_n=np.zeros((I+1)**2)


for L in U:
    R=L[0]
    S=L[1]
    for i in range((I+1)**2):
        q=int((i/(I+1)))
        r=i%(I+1)
        U_n[i] += R[q]*S[r]


U_n = np.reshape(U_n,(I+1,I+1))

fig=plt.figure()
ax = Axes3D(fig)

ax.plot_surface(X,Y,U_n,color="orange")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
