import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import math

#In this part, we implement a discrete solver for the BS equation  with
# Dirichlet boundary conditions, rhs f specified as a function and initial
# condition u0 specified as a function. For the space discretization, we use FEM
# and theta-scheme for the time discretization

# Black-Scholes constants:
sigma = 0.3
T = 1
r = 0.01
R = 4
K = 1
G = R/2



# Functions

# Space discretization:
def tridiag(N,a,b,c):
    diaga = a*np.ones(N-1)
    diagb = b*np.ones(N)
    diagc = c*np.ones(N-1)
    M = sp.diags([diaga,diagb,diagc],[-1,0,1]);
    return M

def build_massMatrix(N):
    h = 2*R/(N+1)
    a = 1./6.
    b = 2./3.
    M = h*tridiag(N,a,b,a)
    return M

def build_rigidityMatrix(N):
    h = 2*R/(N+1)
    a = -1.
    b = 2.
    A = 1./h*tridiag(N,a,b,a)
    return A;

def build_W(N):
    return 1/2*tridiag(N,-1,0,1)

def build_BSMatrix(N):
    M = build_massMatrix(N)
    S = build_rigidityMatrix(N)
    W = build_W(N)
    return sigma**2/2.*S + (sigma**2/2 - r)*W + r*M

def build_F(N):
    return np.zeros(N)

def initialize(N):
    h = 2*R/(N+1)
    xi = np.linspace(-R,R,N+2)
    xi = xi[1:-1]
    g = 0*xi #initialize to zero.
    ind1 = np.where(xi <= np.log(K))
    ind1 = ind1[0]
    ind2 = np.where(xi >= np.log(K))
    ind2 = ind2[0]
    i0 = ind1[-1]
    j0 = ind2[0]
    if i0 == j0:
        g[:i0] = 0
        g[i0] = h/2
        g[i0+1:] = h
    else:
        g[:i0] = 0
        g[i0] = (xi[j0] - np.log(K))**2/(2*h)
        g[j0] = h - (np.log(K) - xi[i0])**2/(2*h)
        g[j0+1:] = h
    M = build_massMatrix(N)
    M = M.todense() # Avoid the CSC warning.
    u0N = np.linalg.solve(M,g)
    return u0N


# Time discretization:
def build_Btheta(M,A,k,theta):
    return M + k*theta*A

def build_Ctheta(M,A,k,theta):
    return M - k*(1 -theta)*A

def build_Ftheta(N,k):
    Ftheta = k*build_F(N)
    return Ftheta;


def bs_formula_C(s,t):
    # Value of an European call option
    if t==0:
        t = 1e-10 # Replace by very small value to avoid runtime warning
    d1 = (np.log(s/K) + (0.5*sigma**2+r)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    P = s*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return P

def bs_formula_P(s,t):
    # Value of an European put option
    if t==0:
        t = 1e-10 # Replace by very small value to avoid runtime warning
    d1 = (np.log(s/K) + (0.5*sigma**2+r)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    P = -s*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    return P
def bs_formula_Pdigital(s,t):
    if t==0:
        t = 1e-10 # Replace by very small value to avoid runtime warning
    d = (np.log(s/K) + (r - 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    P = np.exp(-r*t)*norm.cdf(d)
    return P
# Solver

# function to compute the next iterate
def nextIter(um,B,C,F):
    ump1 = spsolve(B,C@um + F)
    return ump1


def plotOptionValue(exactu,theta,k,T):


    #adjust k.
    M = np.ceil(T/k)
    k = T/M
    m = 0
    um0 = initialize(N)
    um = um0
    Mass = build_massMatrix(N);
    A = build_BSMatrix(N);
    B = build_Btheta(Mass,A,k,theta)
    C = build_Ctheta(Mass,A,k,theta)
    F = build_Ftheta(N,k)

    for m in np.arange(0,M):
            # um is the solution at t = k*m
            um = nextIter(um,B,C,F)
            # Now um is the solution at t = k*(m+1)
    # Now wm is the solution at t = k*([M - 1] + 1) = T.
    TF = T
    fig = plt.figure();
    ax = fig.add_subplot(111)
    xi = np.linspace(-R,R,N+2)
    xi = xi[1:-1]
    ind = np.abs(xi) < G
    uexact_plot = exactu(TF,xi[ind])
    uapprox_plot = um[ind];
    plt.plot(np.exp(xi[ind]),uapprox_plot,'b',label ='FE solution' )
    plt.plot(np.exp(xi[ind]),uexact_plot,'r--',label ='Exact solution' )
    plt.plot(np.exp(xi[ind]),g(xi[ind]),label='Payoff function')
    plt.xlabel('Stock price s')
    plt.ylabel('Option value')
    plt.legend()
    tit = "Digital option value at t=0, maturity T = {T}".format(T = T);
    plt.title(tit)
    plt.show()

# Function to plot error curve
def errorCurve(exactu,theta):
    p1 = 5
    p2 = 12
    exponents = np.arange(p1,p2)

    Ns = 2**exponents
    errL2= np.zeros(len(exponents))
    errH1= np.zeros(len(exponents))
    for p in exponents:
        N = (2**p);
        xi = np.linspace(-R,R,N+2)
        xi = xi[1:-1]
        h = (2.*R)/(N+1)
        k = h
        M = build_massMatrix(N);
        A = build_BSMatrix(N);
        m = 0
        um0 = initialize(N)
        um = um0
        B = build_Btheta(M,A,k,theta)
        C = build_Ctheta(M,A,k,theta)
        F = build_Ftheta(N,k)

        while k*(m+1) < T:
            m = m+1
            um = nextIter(um,B,C,F)
        ind = np.abs(xi) < G
        err = um[ind] - exactu(k*m,xi[ind]);
        aux = M.tocsr()[ind,:]
        MGG = aux.tocsc()[:,ind]
        K = build_rigidityMatrix(N)
        aux = K.tocsr()[ind,:]
        KGG = aux.tocsc()[:,ind]
        errL2[p-p1] = np.sqrt(np.dot(err,MGG@err))
        errH1[p-p1] = np.sqrt(np.dot(err,KGG@err))
    fig, ax = plt.subplots()
    ax.loglog(Ns,errL2,label = 'L2 error')
    ax.loglog(Ns,errH1,label = 'H1 error')
    ax.loglog(Ns,1/(Ns),'g--',label ='O(h)')
    ax.loglog(Ns,10/(Ns**2),'r--',label ='O(h^2)')
    plt.xlabel('Number of grid points N')
    plt.ylabel('L^2 error on (-{G},{G})'.format(G = G))
    plt.title('Convergence in L^2(-{G},{G}), question a), theta = {theta}'.format(G = G,theta =theta))
    plt.ylim((1e-8,1))
    legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.show()


def errorCurveQuestionb(exactu):
    p1 = 5
    p2 = 12
    exponents = np.arange(p1,p2)

    Ns = 2**exponents
    errL2= np.zeros(len(exponents))
    for p in exponents:
        N = (2**p);
        xi = np.linspace(-R,R,N+2)
        xi = xi[1:-1]
        h = (2.*R)/(N+1)
        Mass = build_massMatrix(N);
        A = build_BSMatrix(N);
        m = 0
        um0 = initialize(N)
        um = um0
        # First perform 7 1-scheme time steps with k = h^2
        k = h**2
        theta = 1
        B = build_Btheta(Mass,A,k,theta)
        C = build_Ctheta(Mass,A,k,theta)
        F = build_Ftheta(N,k)
        for m in range(7):
            um = nextIter(um,B,C,F)
        t0 = m*k

        # Use the new intialization for the 1/2 scheme
        # Adjust time step
        M = np.ceil((T-t0)/h)

        k = (T-t0)/M
        theta = 1/2
        m = 0
        B = build_Btheta(Mass,A,k,theta)
        C = build_Ctheta(Mass,A,k,theta)
        F = build_Ftheta(N,k)
        while t0 + k*(m+1) < T:
            m = m+1
            um = nextIter(um,B,C,F)
        ind = np.abs(xi) < G
        err = um[ind] - exactu(k*m,xi[ind]);
        aux = Mass.tocsr()[ind,:]
        MGG = aux.tocsc()[:,ind]
        errL2[p-p1] = np.sqrt(np.dot(err,MGG@err))
    fig, ax = plt.subplots()
    ax.loglog(Ns,errL2,label = 'L2 error')
    ax.loglog(Ns,1/(Ns),'g--',label ='O(h)')
    ax.loglog(Ns,10/(Ns**2),'r--',label ='O(h^2)')
    plt.xlabel('Number of grid points N')
    plt.ylabel('L^2 error on (-{G},{G})'.format(G = G))
    plt.title('Convergence in L^2(-{G},{G}), question b)'.format(G = G))
    plt.ylim((1e-8,1))
    legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.show()


def errorCurveQuestionc(exactu,beta):
    p1 = 5
    p2 = 12
    exponents = np.arange(p1,p2)

    Ns = 2**exponents
    errL2= np.zeros(len(exponents))
    errL2L2 = np.zeros(len(exponents))
    for p in exponents:
        N = (2**p);
        xi = np.linspace(-R,R,N+2)
        xi = xi[1:-1]
        h = (2.*R)/(N+1)
        Mass = build_massMatrix(N);
        A = build_BSMatrix(N);
        m = 0
        um0 = initialize(N)
        um = um0
        theta = 1/2
        M = int(np.ceil(T/h))
        tm = 0
        errL2L2[p - p1] = 0
        ind = np.abs(xi) < G
        aux = Mass.tocsr()[ind,:]
        MGG = aux.tocsc()[:,ind]
        for m in range(M):
            tmp1 = T*((m+1.)/(M+0.))**beta
            km = tmp1 - tm
            B = build_Btheta(Mass,A,km,theta)
            C = build_Ctheta(Mass,A,km,theta)
            F = build_Ftheta(N,km)
            um = nextIter(um,B,C,F)
            tm = tmp1
            errm = um[ind] - exactu(tm,xi[ind]);
            errL2L2[p - p1] = errL2L2[p - p1] + km*h*np.dot(errm,MGG@errm)
        errL2L2[p - p1] = np.sqrt(errL2L2[p - p1])
        errL2[p-p1] = np.sqrt(np.dot(errm,MGG@errm))

    fig, ax = plt.subplots()
    ax.loglog(Ns,errL2L2,label = 'L2 L2 error')
    ax.loglog(Ns,errL2,label = 'L2 error ')
    ax.loglog(Ns,1/(Ns),'g--',label ='O(h)')
    ax.loglog(Ns,10/(Ns**2),'r--',label ='O(h^2)')
    plt.xlabel('Number of grid points N')
    plt.ylabel('Error in L^2(J,L^2(-{G},{G}))'.format(G = G))
    plt.title('L^2(-{G},{G}) and L^2(J;L^2(-{G},{G})) error, question c)'.format(G = G))
    plt.ylim((1e-8,1))
    legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
    plt.show()

# Testing:


def exactu(t,x):
    val = bs_formula_Pdigital(np.exp(x),t)
    return val

def g(x):
    res = 0*x;
    ind = np.where(x > np.log(K))
    res[ind] = 1
    return res

test_text = input ("Enter a number.\n1: Plot option value at t=0 \n2. Plot error curve question a)\n3. Plot error curve question b)\n4. Plot error curve question c)\n")

whatToDo = int(test_text)


if whatToDo==1:
    p_text = input("N = 2** ?\n")
    N = 2**int(p_text)
    theta_text = input ("Enter theta\n")
    theta = float(theta_text)
    T_text = input ("Enter T\n")
    T = float(T_text)
    h = 2*R/(N+1)
    k = h
    plotOptionValue(exactu,theta,k,T)
elif whatToDo==2:
    theta_text = input ("Enter theta\n")
    theta = float(theta_text)
    errorCurve(exactu,theta)
elif whatToDo==3:
    errorCurveQuestionb(exactu)
elif whatToDo==4:
    beta_text = input ("Enter beta\n")
    beta = float(beta_text)
    errorCurveQuestionc(exactu,beta)
else:
    print('Good bye')
