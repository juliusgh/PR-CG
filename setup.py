
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import diags, csr_matrix

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