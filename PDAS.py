import numpy as np

def PDAS(A,b,c,x0,lambda0):
    k1 = 1
    tol = 1e-12
    kmax = 50
    #initialize algorithm
    n = b.size
    lamb = lambda0
    k = 1
    Ik = lamb + k1*(c-x0) <= 0
    Ak = lamb + k1*(c-x0)> 0
    B = -np.eye(n)
    C = np.diag(Ak*1.0)
    D = np.diag(Ik*1.0)
    L1 = np.hstack((A,B))
    L2 = np.hstack((C,D))
    LHS = np.vstack((L1,L2))
    RHS = np.hstack((b,C@c))
    y = np.linalg.solve(LHS,RHS)
    x = y[:n]
    lamb = y[n:]
    crit = (np.sqrt(np.dot(x-x0,x-x0))) < tol
    while ((not crit) and (k < kmax)):
        x0 = x
        Ik = lamb + k1*(c-x) <= 0
        Ak = lamb + k1*(c-x) > 0
        C = np.diag(Ak*1.0)
        D = np.diag(Ik*1.0)
        L1 = np.hstack((A,B))
        L2 = np.hstack((C,D))
        LHS = np.vstack((L1,L2))
        RHS = np.hstack((b,C@c))
        y = np.linalg.solve(LHS,RHS)
        x = y[:n]
        lamb = y[n:]
        k = k+1
        crit = (np.sqrt(np.dot(x-x0,x-x0))) < tol
    if crit:
        print('PDAS converged in {k} iterations'.format(k = k))
    else:
        print('Warning: PDAS did not converge')
    return x,lamb


def testPDAS():
    # Size of the test:
    N = 15
    a1 = np.ones(N)
    a2 = -np.ones(N-1)*1/3
    A = np.diag(a1) + np.diag(a2,k=-1) + np.diag(a2,k=1)
    # Same test as PSOR
    u = np.random.rand(N)
    # Prepare test data
    x = u.copy()
    x[u < 0.5] = 0
    b = A@x
    b[u<0.5] -=u[u<0.5]
    c = np.zeros(N)
    x0 = c
    lamb0 = c
    (xguess,lambGuess) = PDAS(A,b,c,x0,lamb0)
    print('x\n',x)
    print('xguess\n',xguess)
    # In theory, x and xguess should be very close.
