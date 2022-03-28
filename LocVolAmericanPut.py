import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import math

import PDAS # For this line to work, PDAS.py must be in the same directory as this script.
PDAS.testPDAS()


# Computation of the value of an american option under a local volatility model

# Put option payoff (global def)
def gput(x):
    S = np.exp(x)
    val = np.maximum(K - S,0)
    return val

def xgrid(N):
    return np.linspace(-R,R,N+2)
def hval(N):
    return 2*R/(N+1)

def Mval(N,T):
    kapprox = hval(N)
    return int(np.ceil(T/kapprox))

def kval(N,T):
    print('M',Mval(N,T))
    return T/Mval(N,T)

def tgrid(N,T):
    return kval(N,T)*np.arange(0,Mval(N,T)+1)

# FE assembling routines

R = 3 # Truncation parameter
leftLim = -R/2
rightLim = R/6 # Area of interest for plots

def assembleMatrixA(N,alpha):
    xi = xgrid(N)
    h = hval(N)

    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    diag = 1/(2*h)*(alpha(xi[:-2] + p1) + alpha(xi[:-2] +p2))
    diag += 1/(2*h)*(alpha(xi[1:-1]+p1) + alpha(xi[1:-1]+p2))
    lodiag = -1/(2*h)*(alpha(xi[1:-2]+p1) + alpha(xi[1:-2]+p2))
    updiag = lodiag
    A = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return A


def assembleMatrixB(N,beta):
    xi = xgrid(N)
    h = hval(N)

    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = p1/h
    w2 = p2/h
    diag = 1/2*(beta(xi[:-2] + p1)*w1 + beta(xi[:-2] + p2)*w2)
    diag += -1/2*(beta(xi[1:-1] + p1)*w2 + beta(xi[1:-1] + p2)*w1)
    lodiag = -1/2*(beta(xi[1:-2] + p1)*w1 + beta(xi[1:-2] + p2)*w2)
    updiag =  1/2*(beta(xi[1:-2] + p1)*w2 + beta(xi[1:-2] + p2)*w1)
    B = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return B

def assembleMatrixC(N,gamma):
    xi = xgrid(N)
    h = hval(N)

    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = p1/h
    w2 = p2/h

    diag = h/2*(gamma(xi[:-2]+p1)*w1**2 + gamma(xi[:-2]+p2)*w2**2)
    diag += h/2*(gamma(xi[1:-1]+p1)*w2**2 + gamma(xi[1:-1]+p2)*w1**2)
    updiag = h/2*w1*w2*(gamma(xi[1:-2]+p1)+gamma(xi[1:-2]+p2))
    lodiag = updiag
    C = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return C

def assembleMatrix(N,alpha,beta,gamma):
    A = assembleMatrixA(N,alpha)
    B = assembleMatrixB(N,beta)
    C = assembleMatrixC(N,gamma)
    return A + B + C

def assembleMass(N):
    def alpha(x):
        return 0*x
    def beta(x):
        return 0*x
    def gamma(x):
        return 0*x+1
    return assembleMatrix(N,alpha,beta,gamma)

def assembleALV(N,tilde_sigma,dtilde_sigma,r):
    def alpha(x):
        return tilde_sigma(x)**2/2
    def beta(x):
        return tilde_sigma(x)*dtilde_sigma(x) + alpha(x) - r
    def gamma(x):
        return 0*x + r
    return assembleMatrix(N,alpha,beta,gamma)

def plotFE(vec,lab):
    N = vec.size
    xi = xgrid(N)
    vals = 0*xi
    vals[1:-1] = vec
    ind = (xi > leftLim)*(xi < rightLim)
    plt.plot(np.exp(xi[ind]),vals[ind],label=lab)

# Solver
def nextIter(um,A,Fm,lamb):
    c = np.zeros(um.size)
    (ump1,lamb) = PDAS.PDAS(A.todense(),Fm,c,um,lamb)
    return (ump1,lamb)

def computeOptionValue(r,tilde_sigma,dsigma_tilde,K,T,N):
    xi = xgrid(N)
    M = Mval(N,T)
    k = kval(N,T)
    Mass = assembleMass(N)
    ALV = assembleALV(N,tilde_sigma,dtilde_sigma,r)
    A = Mass + k*ALV
    G = -ALV*gput(xi[1:-1])

    um = np.zeros(N)
    lamb = np.zeros(N)

    for m in range(1,M+1):
        Fm = Mass@um + k*G
        (um,lamb) = nextIter(um,A,Fm,lamb)

    dof = xi[1:-1]
    vm = um + gput(dof) # um is in "excess to payoff", vm is the option value
    return vm


def computeExBoundary(r,tilde_sigma,dsigma_tilde,K,T,N):
    xi = xgrid(N)
    M = Mval(N,T)
    k = kval(N,T)
    Mass = assembleMass(N)
    ALV = assembleALV(N,tilde_sigma,dtilde_sigma,r)
    A = Mass + k*ALV
    G = -ALV*gput(xi[1:-1])

    um = np.zeros(N)
    lamb = np.zeros(N)
    exBound = 0*tgrid(N,T)
    exBound[0] = K
    for m in range(1,M+1):
        Fm = Mass@um + k*G
        (um,lamb) = nextIter(um,A,Fm,lamb)
        ind = np.where(um > 1e-5)
        exBound[m] = np.exp(xi[ind[0][0]])
    return exBound


N = 400

# Definition of the local volatility
r = 0.05
sig = 0.3
def sigma(s):
    # return sig*(1 + 0.5*np.sin(np.log(s+1)))
    return sig*(1.2 + 0.5*np.cos(22*np.log(s+1)))

def d_sigma(s):
    # return sig*0.5*np.cos(np.log(s+1))*1/(s+1)
    return -sig*(11*np.sin(np.log(s+1))*1/(s+1))

def tilde_sigma(x):
    return sigma(np.exp(x))

def dtilde_sigma(x):
    return d_sigma(np.exp(x))*np.exp(x)

T = 1
K = 1


# Computation of the option value
Vput = computeOptionValue(r,tilde_sigma,dtilde_sigma,K,T,N)

#Visualization
plotFE(Vput,'American put option')
xi = xgrid(N)
ind = (xi > leftLim)*(xi < rightLim)
plt.plot(np.exp(xi[ind]),gput(xi[ind]),'k--',label='payoff')
plt.plot(np.exp(xi[ind]),tilde_sigma(xi[ind]),'r--',label='local volatility')
plt.xlabel('spot price')
plt.ylabel('Option value')
plt.title('American put option in local volatility model')
plt.legend()
plt.show()


#Computation of the exercise boundary
tm = tgrid(N,T)
exBound = computeExBoundary(r,tilde_sigma,dtilde_sigma,K,T,N)

#Visualization
plt.plot(tm,exBound)
plt.xlabel('T-t')
plt.ylabel('s^*(t)')
plt.title('Exercise boundary')
plt.show()
