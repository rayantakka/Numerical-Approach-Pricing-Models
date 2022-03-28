import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import math

# Implementation of the PSOR algorithm


def criterion(A,b,c,x,tol):
    u1 = x-c
    u2 = np.squeeze(np.array(A@x - b))

    critOrtho = np.abs(np.dot(u1,u2)) < tol
    crit1 = np.all(u1 > -tol)
    crit2 = np.all(u2 > -tol)
    return (critOrtho and crit1 and crit2)

def PSOR(A,b,c,x0):
    # Algotirhm Parameters
    omega = 1.5
    epsilon = 1e-12
    maxit = 1000

    # Preparation
    N= A.shape
    N = N[1]
    L = np.tril(A,k=-1)
    U = np.triu(A,k=1)
    D = A - L - U

    # Initialization
    xk = x0
    crit= criterion(A,b,c,xk,epsilon)
    it = 0
    while ((not crit) and (it < maxit)):
        it = it+1
        xkp1 = xk.copy()
        y = np.zeros(N)
        for i in range(N):
            y[i] = 1/A[i,i]*(b[i] - np.dot(L[i,:],xkp1) - np.dot(U[i,:],xk))
            xkp1[i] = np.maximum(c[i],xk[i] + omega*(y[i] - xk[i]))
        # crit = np.sqrt(np.dot(xk - xkp1,xk - xkp1)) < epsilon
        xk = xkp1.copy()
        crit= criterion(A,b,c,xk,epsilon)
    if not crit:
        print('Warning: PSOR did not converge')
    else:
        print('Converged in {it} iterations'.format(it = it))
    return xk


# Test the algorithm

def testPSOR():
    # Size of the test:
    N = 15
    a1 = np.ones(N)
    a2 = -np.ones(N-1)*1/3
    A = np.diag(a1) + np.diag(a2,k=-1) + np.diag(a2,k=1)
    # This is a diagonally dominant, positive definite matrix, so the algorithm
    # must converge in theory.
    u = np.random.rand(N)
    # Prepare test data
    x = u.copy()
    x[u < 0.5] = 0
    b = A@x
    b[u<0.5] -=u[u<0.5]
    c = np.zeros(N)
    x0 = c
    xguess = PSOR(A,b,c,x0)
    print('x\n',x)
    print('xguess\n',xguess)
    # In theory, x and xguess should be very close.
