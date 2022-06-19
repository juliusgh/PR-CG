import numpy as np
import scipy as sp
from scipy import sparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
import time
#import psutil
import cg
import pr_cg

def setup_1d_poisson(n):
    # compute mesh width from n (number of interior points)
    h = 1 / (n+1);
    nn = n;
    # assemble coefficient matrix A
    k = [-np.ones(n-1),2*np.ones(n),-np.ones(n-1)]
    offset = [-1,0,1]
    A = diags(k,offset).toarray()
    # initialise remaining vectors
    x_exact = np.zeros(n);
    b = np.zeros(n);
    # compute RHS and exact solution (inner points only)
    for i in range(n):
        xi = (i+1)*h;
        b[i] = -2  + 12*xi - 12*xi**2;
        b[i] = b[i] * (h*h);
        x_exact[i] = (xi*(1-xi))**2;
    x0 = np.zeros(A.shape[0])
    return A, b, x_exact, x0

def setup_2d_poisson(n):
    # compute mesh width from n (number of interior points per dimension)
    h = 1 / (n+1);
    nn = n*n;
    # assemble coefficient matrix A, quick and dirty 
    # see https://de.mathworks.com/help/matlab/ref/kron.html
    I = sparse.eye(n,n)
    E = csr_matrix((np.ones(n-1),(np.arange(1,n),np.arange(0,n-1))),shape=(n,n)) #???
    D = E+E.T-2*I;
    A = -(sparse.kron(D,I)+sparse.kron(I,D));
    # initialise remaining vectors
    x_exact = np.zeros(n*n);
    x0 = np.zeros(n*n);
    b = np.zeros(n*n);
    # compute RHS and exact solution (inner points only)
    for i in range(1,n+1):
        for j in range(1,n+1):
            xij = i*h; yij = j*h
            b[(j-1)*n+i-1] = -2*(6*xij**2-6*xij+1)*(yij-1)**2*yij**2 - 2*(xij-1)**2*xij**2*(6*yij**2-6*yij+1)
            b[(j-1)*n+i-1] = b[(j-1)*n+i-1] * (h*h)
            x_exact[(j-1)*n+i-1] = (xij*(1-xij)*yij*(1-yij))**2
    return A, b, x_exact, x0

def setup_matrix_market(matrix_name):
    A = csr_matrix(sp.io.mmread(f"matrices/{matrix_name}.mtx"))
    N = A.get_shape()[0]
    x_exact = np.ones(N) / np.sqrt(N)
    b = A @ x_exact
    x0 = np.zeros(A.shape[0])
    return A, b, x_exact, x0

def setup_2d_convdiff(n, central_differences=True):
    h = 1 / (n+1)
    nn = n*n

    a = lambda x,y:  20*np.exp(3.5*(x**2 + y**2))
    dadx = lambda x,y: 140*x*np.exp(3.5*(x**2 + y**2))

    A = csr_matrix((n**2,n**2))

    for i in range(1,n+1):
        xij = i*h; # x coordinate
    for j in range(1,n+1):
        yij = j*h; # y coordinate
        k = (j-1)*n + i; # center index (linear)
        kiP = (j-1)*n + i + 1; # i+1
        kiM = (j-1)*n + i - 1; # i-1
        kjP = j*n + i; # j+1
        kjM = (j-2)*n + i; # j-1

        # center part of FD stencil (x & y) : - 4 u(i,j)
        A[k-1,k-1] = A[k-1,k-1] + 4;

        # d^2/dy^2 \approx ( u(i,j+1) - 2 u(i,j) + u(i,j-1) ); - 2 u(i,j) included in center part
        if j < n:
            A[k-1,kjP-1] = A[k-1,kjP-1] - 1
        
        if j>1:
            A[k-1,kjM-1] = A[k-1,kjM-1] - 1

        # d^2/dx^2 \approx ( u(i+1,j) - 2 u(i,j) + u(i-1,j) ); - 2 u(i,j) included in center part
        if i < n:
            A[k-1,kiP-1] = A[k-1,kiP-1] - 1
        
        if i > 1:
            A[k-1,kiM-1] = A[k-1,kiM-1] - 1

        # 0.5*da/dx u
        A[k-1,k-1] = A[k-1,k-1] + 0.5 * dadx(xij,yij)*(h**2)

        # a du / dx
        if central_differences: # central differences
            if i < n:
                A[k-1,kiP-1] = A[k-1,kiP-1] + a(xij,yij)*(h/2)
                
            if i > 1:
                A[k-1,kiM-1] = A[k-1,kiM-1] - a(xij,yij)*(h/2)
                
        else: # backward differences
            if i > 1:
                A[k-1,kiM-1] = A[k-1,kiM-1] - a(xij,yij)*(h)
                
            A[k-1,k-1] = A[k-1,k-1] + a(xij,yij)*(h)
            

    b = np.zeros(n**2);
    x_exact = np.zeros(n**2);
    # exact function (x(1-x)y(1-y))^2
    for i in range(1,n+1):
        xij = i*h;
        for j in range(1,n+1):
            yij = j*h;
            k = (j-1)*n + i; # linear index
            # the following can also be done by b = A*x; b*=(h*h)
            u = (xij*(1-xij)*yij*(1-yij))**2;
            dudx = 2*yij**2*(1-yij)**2*xij*(1-xij)*(1 - 2*xij)
            #dudy = 2*xij**2*(1-xij)**2*yij*(1-yij)*(1-2*yij);
            dudxx = 2*yij**2 * (1-yij)**2 * ( 6*xij**2 - 6*xij + 1)
            dudyy = 2*xij**2*(1-xij)**2*( 6*yij**2 - 6*yij + 1)

            b[k-1] = - dudxx - dudyy + 0.5*dadx(xij,yij) * u + a(xij,yij) * dudx
            b[k-1] = b[k-1] * (h*h)

            x_exact[k-1] = u

    x0 = np.zeros(n**2)
    return A, b, x_exact, x0

#N_physical_cores = psutil.cpu_count(logical=False)
#N_logical_cores = psutil.cpu_count(logical=True)
#print(f"The number of physical/logical cores is {N_physical_cores}/{N_logical_cores}")

# model problems
n = 64
#A, b, x_exact, x0 = setup_1d_poisson(n)
A, b, x_exact, x0 = setup_2d_poisson(n)
#A, b, x_exact, x0 = setup_matrix_market('bcsstk03') #'model_48_8_3')bcsstk03,nos2
#A, b, x_exact, x0 = setup_2d_convdiff(n)
#print(A.todense())
dtype = np.float64
A = A.astype(dtype)
b = b.astype(dtype)
x0 = x0.astype(dtype)

# preconditioners
jacobi = lambda x: (1/A.diagonal())*x

start = time.process_time()
x = sparse.linalg.spsolve(A, b)
print(time.process_time() - start)
print('residual norm:', np.linalg.norm(A @ x - b))

start = time.process_time()
x1, iterates1, residuals1 = cg.pcg(A, b, x0, x_exact, preconditioner=jacobi, max_iter=1200, eps=0)
print(x1.dtype)
print(time.process_time() - start, len(iterates1))
print('residual norm:', np.linalg.norm(A @ x1 - b))

start = time.process_time()
x2, iterates2, residuals2 = cg.pr_pcg(A, b, x0, x_exact, preconditioner=jacobi, max_iter=1200, eps=0, recompute=True)
print(x1.dtype)
print(time.process_time() - start, len(iterates1))
print('residual norm:', np.linalg.norm(A @ x2 - b))

start = time.process_time()
x3, iterates3, residuals3 = cg.m_pcg(A, b, x0, x_exact, preconditioner=jacobi, max_iter=1200, eps=0, recompute=True)
#x3, iterates3, residuals3 = cg.pr_master_pcg(A, b, x0, preconditioner=jacobi, max_iter=1200, eps=0)
print(time.process_time() - start, len(iterates2))
print('residual norm:', np.linalg.norm(A @ x3 - b))

error1 = iterates1 - x_exact
error2 = iterates2 - x_exact
error3 = iterates3 - x_exact
print(A.shape, error1.shape)
e1_2norm = np.linalg.norm(error1, axis=1)
e2_2norm = np.linalg.norm(error2, axis=1)
e3_2norm = np.linalg.norm(error3, axis=1)
#e1_2norm = np.apply_along_axis(lambda e: np.linalg.norm(e), 1, error1)
#e2_2norm = np.apply_along_axis(lambda e: np.linalg.norm(e), 1, error2)
e1_Anorm = np.apply_along_axis(lambda e: np.sqrt(e.T @ A @ e), 1, error1)
e2_Anorm = np.apply_along_axis(lambda e: np.sqrt(e.T @ A @ e), 1, error2)
r1_2norm = np.linalg.norm(residuals1, axis=1)
r2_2norm = np.linalg.norm(residuals2, axis=1)
#plt.semilogy(np.arange(1,len(r1_2norm)+1), r1_2norm, label='residual1')
#plt.semilogy(np.arange(1,len(r2_2norm)+1), r2_2norm, label='residual2')
plt.semilogy(np.arange(1,len(e1_2norm)+1), e1_2norm, label='error1_2norm')
plt.semilogy(np.arange(1,len(e2_2norm)+1), e2_2norm, label='error2_2norm')
plt.semilogy(np.arange(1,len(e3_2norm)+1), e3_2norm, label='error3_2norm')
#plt.semilogy(np.arange(1,len(e1_2norm)+1), e1_Anorm, label='error1_Anorm')
#plt.semilogy(np.arange(1,len(e2_2norm)+1), e2_Anorm, label='error2_Anorm')
#plt.semilogy(np.arange(1,len(am1)+1), am1, label='alpha1')
plt.legend()
plt.show()
plt.spy(A)
plt.show()