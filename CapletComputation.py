import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import math


# Computation of the value of a caplet.

def rgrid(N):
    return np.linspace(0,R,N+2)
def hval(N):
    return R/(N+1)

def Mval(N,T):
    kapprox = hval(N)
    return int(np.ceil(T/kapprox))

def kval(N,T):
    return T/Mval(N,T)

def tgrid(N,T):
    return kval(N,T)*np.arange(0,Mval(N,T)+1)

# FE assembling routines

R = 4 # Truncation parameter
leftLim = 0
rightLim = R/2 # Area of interest for plots

def assembleMatrixA(N,A):
    ri = rgrid(N)
    h = hval(N)
    dof = ri[:-1]
    diag = np.zeros(N+1)
    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    diag[0] = 1/(2*h)*(A(p1) + A(p2))
    diag[1:] = 1/(2*h)*(A(dof[:-1] + p1) + A(dof[:-1] +p2))
    diag[1:] += 1/(2*h)*(A(dof[1:]+p1) + A(dof[1:]+p2))
    lodiag = -1/(2*h)*(A(dof[:-1]+p1) + A(dof[:-1]+p2))
    updiag = lodiag
    M = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return M

def assembleMatrixB(N,B):
    ri = rgrid(N)
    h = hval(N)
    dof = ri[:-1]
    diag = np.zeros(N+1)
    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = p1/h
    w2 = p2/h
    diag[0] = -1/2*(B(p1)*w2 + B(p2)*w1)
    diag[1:] = 1/2*(B(dof[:-1] + p1)*w1 + B(dof[:-1] + p2)*w2)
    diag[1:] += -1/2*(B(dof[1:] + p1)*w2 + B(dof[1:] + p2)*w1)
    lodiag = -1/2*(B(dof[:-1] + p1)*w1 + B(dof[:-1] + p2)*w2)
    updiag =  1/2*(B(dof[:-1] + p1)*w2 + B(dof[:-1] + p2)*w1)
    M = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return M

def assembleMatrixC(N,C):
    ri = rgrid(N)
    h = hval(N)
    dof = ri[:-1]
    diag = np.zeros(N+1)
    p1 = h/2 - h/(2*np.sqrt(3))
    p2 = h/2 + h/(2*np.sqrt(3))
    w1 = p1/h
    w2 = p2/h
    diag[0] = h/2*(C(p1)*w2**2+C(p2)*w1**2)
    diag[1:] = h/2*(C(dof[:-1]+p1)*w1**2 + C(dof[:-1]+p2)*w2**2)
    diag[1:] += h/2*(C(dof[1:]+p1)*w2**2 + C(dof[1:]+p2)*w1**2)
    updiag = h/2*w1*w2*(C(dof[:-1]+p1)+C(dof[:-1]+p2))
    lodiag = updiag
    M = sp.diags([lodiag,diag,updiag],[-1,0,1])
    return M

def assembleMatrix(N,A,B,C):
    M1 = assembleMatrixA(N,A)
    M2 = assembleMatrixB(N,B)
    M3 = assembleMatrixC(N,C)
    return M1 + M2 + M3

def assembleMass(N,mu):
    def A(r):
        return 0*r
    def B(r):
        return 0*r
    def C(r):
        return r**(2*mu)
    return assembleMatrix(N,A,B,C)

def assembleAcaplet(N,alpha,beta,sigma,mu):
    def A(r):
        return sigma**2/2*r**(1 + 2*mu)
    def B(r):
        return (1/2 + mu)*sigma**2*r**(2*mu)  + (beta*r - alpha)*r**(2*mu)
    def C(r):
        return r**(1 + 2*mu)
    return assembleMatrix(N,A,B,C)

def plotFE(vec,lab):
    N = vec.size
    ri = rgrid(N)
    vals = 0*xi
    vals[1:-1] = vec
    ind = (ri > leftLim)*(ri < rightLim)
    plt.plot(np.exp(xi[ind]),vals[ind],label=lab)

# Solver
def nextIter(um,B,C):
    ump1 = spsolve(B,C@um)

def computeBond(sigma,alpha,beta,mu,T,N,theta):
    ri = rgrid(N)
    M = Mval(N,T)
    k = kval(N,T)
    Mass = assembleMass(N,mu)
    Mass = Mass[1:,1:]

    Acaplet = assembleAcaplet(N,alpha,beta,sigma,mu)
    Acaplet = Acaplet[1:,1:]
    B = Mass + theta*k*Acaplet
    C = Mass - (1 - theta)*k*Acaplet

    Vm = np.zeros(N)+1

    for m in range(1,M+1):
        Vm = spsolve(B,C@Vm)

    return Vm

def computeCaplet(sigma,alpha,beta,mu,T,T1,N,theta,g):
    M = Mval(N,T)
    k = kval(N,T)

    Vbond = computeBond(sigma,alpha,beta,mu,T1-T,N,theta)
    Vcaplet = g(Vbond)

    Mass = assembleMass(N,mu)
    Mass = Mass[1:,1:]
    Acaplet = assembleAcaplet(N,alpha,beta,sigma,mu)
    Acaplet = Acaplet[1:,1:]
    B = Mass + theta*k*Acaplet
    C = Mass - (1 - theta)*k*Acaplet
    for m in range(1,M+1):
        Vcaplet = spsolve(B,C@Vcaplet)


    return (Vcaplet,Vbond)

# Computation of the option value
N = 512

# Definition of the local volatility
sigma = 0.1
K = 0.05
mu = -0.1
T1 = 4
T = 2
alpha = 0.025
beta = 0.5
theta = .5

def g(v):
    return (T1 -T)*v*np.maximum(((1-v)/((T1-T)*v) - K),0)

(Vcaplet,Vbond) = computeCaplet(sigma,alpha,beta,mu,T,T1,N,theta,g)


ri = rgrid(N)
ri = ri[1:-1]
ind = (0.0 <ri) & (ri < 0.2)
plt.plot(ri[ind],Vcaplet[ind],label='Caplet price')
plt.plot(ri[ind],g(Vbond[ind]),'--',label='g(Vbond)')
plt.xlabel('Interest rate')
plt.ylabel('Price')
plt.title('Price of a Caplet')
plt.legend()
plt.show()
