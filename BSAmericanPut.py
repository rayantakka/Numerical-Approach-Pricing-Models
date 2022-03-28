import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import math

import PSOR # For this line to work, PSOR.py must be in the same directory as this script.


# Computation of the value of an american put option and plot of the exercise boundary

PSOR.testPSOR() # Calls testPSOR() which is defined in PSOR.py.


R = 3
leftLim = -R/2
rightLim = R/2

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
# Explicit B-S formulas:

def bs_formula_C(r,sigma,s,t,K):
    # Value of an European call option
    if t==0:
        t = 1e-10 # Replace by very small value to avoid runtime warning
    d1 = (np.log(s/K) + (0.5*sigma**2+r)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    P = s*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return P


def bs_formula_P(r,sigma,s,t,K):
    # Value of an European put option
    if t==0:
        t = 1e-10 # Replace by very small value to avoid runtime warning
    d1 = (np.log(s/K) + (0.5*sigma**2+r)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    P = -s*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    return P

# FEM assembling routines

def tridiag(N,a,b,c):
    diaga = a*np.ones(N-1)
    diagb = b*np.ones(N)
    diagc = c*np.ones(N-1)
    M = sp.diags([diaga,diagb,diagc],[-1,0,1]);
    return M

def build_massMatrix(N):
    h = hval(N)
    a = 1./6.
    b = 2./3.
    M = h*tridiag(N,a,b,a)
    return M

def build_rigidityMatrix(N):
    h = hval(N)
    a = -1.
    b = 2.
    A = 1./h*tridiag(N,a,b,a)
    return A;

def build_W(N):
    return 1/2*tridiag(N,-1,0,1)

def build_BSMatrix(N,r,sigma):
    M = build_massMatrix(N)
    S = build_rigidityMatrix(N)
    W = build_W(N)
    return sigma**2/2.*S + (sigma**2/2 - r)*W + r*M


def build_G(N,r,sigma,K):
    # Compute Fi = -a^{BS}(g,bi)
    xi = xgrid(N)
    h = hval(N)
    xi = xi[1:-1]
    G = 0*xi
    ind1 = np.where(xi <= np.log(K))
    ind1 = ind1[0]
    ind2 = np.where(xi >= np.log(K))
    ind2 = ind2[0]
    i0 = ind1[-1]
    j0 = ind2[0]
    # log(K) is in [xi0,xj0]
    G[:i0] = -r*K*h
    G[j0+1:] = 0
    if i0==j0:
        G[j0] = sigma**2*K/2 - r*K*h/2
    else:
        s = xi[j0] - np.log(K)
        p = s/h
        G[i0] = 1/2*sigma**2*K*p -r*K/2*(2*h - p*s)

        q = np.log(K) - xi[i0]
        p = q/h
        G[j0] =  1/2*sigma**2*K*p -r*K*q*p/2
    return G


# Solver
def nextIter(um,B,F):
    c = 0*F
    ump1 = PSOR.PSOR(B.todense(),F,c,um)
    return ump1

def computeOptionValue(r,sigma,K,T,N):
    xi = xgrid(N)
    h = hval(N)
    # Prepare matrices for iteration:
    Mass = build_massMatrix(N)
    ABS = build_BSMatrix(N,r,sigma)
    M = Mval(N,T)
    k = kval(N,T)
    B = Mass + k*ABS
    G = build_G(N,r=r,sigma=sigma,K=K)
    #Initialize:
    um = np.zeros(N)
    Fm = k*G + Mass*um
    # Iterate
    for m in range(1,M+1):
        um = nextIter(um,B,Fm)
        Fm = k*G + Mass*um
        tm = k*m
    print('tm',tm)
    payoff = gput(xi[1:-1])
    return um + payoff

def computeExBoundary(r,sigma,K,T,N):
    xi = xgrid(N)
    tm = tgrid(N,T)
    h = hval(N)
    k = kval(N,T)
    # Prepare matrices for iteration:
    Mass = build_massMatrix(N)
    ABS = build_BSMatrix(N,r,sigma)
    B = Mass + k*ABS
    G = build_G(N,r=r,sigma=sigma,K=K)
    #Initialize:
    um = np.zeros(N)
    Fm = k*G + Mass@um
    # Iterate
    exBound = 0*tm
    exBound[0] = K
    for m in range(1,Mval(N,T)+1):
        um = nextIter(um,B,Fm)
        Fm = k*G + Mass@um
        ind = np.where(um > 1e-5)
        exBound[m] = np.exp(xi[ind[0][0]])
    return exBound


# Plotting routine
def plotFE(vec,lab):
    N = vec.size
    xi = xgrid(N)
    vals = 0*xi
    vals[1:-1] = vec
    ind = (xi > leftLim)*(xi < rightLim)
    plt.plot(np.exp(xi[ind]),vals[ind],label=lab)


# Black-Scholes model parameters:

r = 0.05 # Riskless interest rate
sigma = 0.3 # Underlying volatility
K = 1 # Strike price
T = 1 # Maturity
N = 401 # Grid size

def gput(x):
    S = np.exp(x)
    val = np.maximum(K - S,0)
    return val

# Computation of the option value
Vamput = computeOptionValue(r=r,sigma=sigma,K=K,T=T,N=N)

# Visualization
xi = xgrid(N)
S = np.exp(xi)
ind = (xi > leftLim)*(xi < rightLim)
plotFE(Vamput,'American put')
plt.plot(S[ind],gput(xi[ind]),'k--',label='payoff')
plt.plot(S[ind],bs_formula_P(r=r,sigma=sigma,s=S[ind],t=T,K=K),'r--',label='European put')
plt.legend()
plt.xlabel('Stock price s')
plt.ylabel('Option value')
plt.title('American vs European put option')
plt.show()

# Computation of the boundary
tm = tgrid(N,T)
exBound = computeExBoundary(r=r,sigma=sigma,K=K,T=T,N=N)

#Visualization
plt.plot(tm,exBound)
plt.xlabel('T-t')
plt.ylabel('s^*(t)')
plt.title('Exercise boundary')
plt.show()
