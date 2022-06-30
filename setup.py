#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Julius Herb (st160887@stud.uni-stuttgart.de)
This module contains functionality to assemble various model problems.
"""

import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.sparse import diags, csr_matrix


def poisson_1d(n, dtype=np.float64):
    """Assemble the model problem "Poisson 1D"

    :param n: Problem size
    :type n: int
    :param dtype: data type of the matrices and vectors, defaults to np.float64
    :type dtype: NumPy data type, optional
    :return: (A, b, x_exact, x0)
    :rtype: tuple of NumPy arrays
    """
    h = 1 / (n+1)
    k = [-np.ones(n-1),2*np.ones(n),-np.ones(n-1)]
    offset = [-1,0,1]
    A = csr_matrix(diags(k,offset))
    b = np.zeros(n)
    for i in range(n):
        xi = (i+1)*h;
        b[i] = -2  + 12*xi - 12*xi**2
        b[i] = b[i] * (h*h)
    x_exact = np.linalg.solve(A.toarray(), b)
    x0 = np.zeros(n)
    
    A = A.astype(dtype)
    b = b.astype(dtype)
    x0 = x0.astype(dtype)
    return A, b, x_exact, x0


def poisson_2d(n, dtype=np.float64):
    """Assemble the model problem "Poisson 2D"

    :param n: Problem size
    :type n: int
    :param dtype: data type of the matrices and vectors, defaults to np.float64
    :type dtype: NumPy data type, optional
    :return: (A, b, x_exact, x0)
    :rtype: tuple of NumPy arrays
    """
    h = 1 / (n + 1)
    N = n * n
    I = sparse.eye(n,n)
    E = csr_matrix((np.ones(n-1),(np.arange(1,n),np.arange(0,n-1))),shape=(n,n))
    D = E + E.T - 2 * I
    A = -(sparse.kron(D,I) + sparse.kron(I,D))
    b = np.zeros(N)
    for i in range(1,n+1):
        for j in range(1,n+1):
            xij = i*h; yij = j*h
            b[(j-1)*n+i-1] = -2*(6*xij**2-6*xij+1)*(yij-1)**2*yij**2 - 2*(xij-1)**2*xij**2*(6*yij**2-6*yij+1)
            b[(j-1)*n+i-1] = b[(j-1)*n+i-1] * (h*h)
    x_exact = np.linalg.solve(A.toarray(), b)
    x0 = np.zeros(N)

    A = A.astype(dtype)
    b = b.astype(dtype)
    x0 = x0.astype(dtype)
    return A, b, x_exact, x0


def matrix_market(matrix_name, dtype=np.float64):
    """Load a model problem from the NIST Matrix Market (https://math.nist.gov/MatrixMarket/)
    Various matrices from the Matrix Market are stored as mtx-files in the directory `/matrices`

    :param matrix_name: Name of the matrix from the Matrix Market in lowercase
    :type matrix_name: str
    :param dtype: data type of the matrices and vectors, defaults to np.float64
    :type dtype: NumPy data type, optional
    :return: (A, b, x_exact, x0)
    :rtype: tuple of NumPy arrays
    """
    A = csr_matrix(mmread(f"matrices/{matrix_name}.mtx"))
    N = A.get_shape()[0]
    x_exact = np.ones(N) / np.sqrt(N)
    b = A @ x_exact
    x0 = np.zeros(A.shape[0])

    A = A.astype(dtype)
    b = b.astype(dtype)
    x0 = x0.astype(dtype)
    return A, b, x_exact, x0

poisson_2d(2)