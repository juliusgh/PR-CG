#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Julius Herb (st160887@stud.uni-stuttgart.de)
This module contains various preconditioned CG variants
using the Predict-and-Recompute technique as well as pipelining.
"""

import numpy as np


def cg(A, b, x0, max_iter=10000, eps=1e-8, variant='cg'):
    """CG method

    Following CG variants are available:
    'cg': vanilla CG method
    'p_cg': Predict-and-Recompute CG method without recomputation
    'pr_cg': Predict-and-Recompute CG method
    'mp_cg': Meurant CG method without recomputation
    'mpr_cg': Meurant CG method
    'pipe_p_cg': Pipelined Predict-and-Recompute CG method without recomputation
    'pipe_pr_cg': Pipelined Predict-and-Recompute CG method
    'pipe_mp_cg': Pipelined Meurant CG method without recomputation
    'pipe_mpr_cg': Pipelined Meurant CG method

    :param A: system matrix A
    :type A: NumPy array
    :param b: right hand side vector b
    :type b: NumPy array
    :param x0: initial guess for the solution
    :type x0: NumPy array
    :param max_iter: maximum number of iterations, defaults to 10000
    :type max_iter: int, optional
    :param eps: epsilon for stopping criterion, defaults to 1e-8
    :type eps: float, optional
    :param variant: cg method variant, defaults to 'cg'
    :type eps: str, optional
    """    
    r0 = b - A @ x0
    r = r0
    p = r
    alpha = np.dot(r, r)
    if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
        v = A @ p

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            u = A @ v
            w = A @ r
        else:
            v = A @ p
        mu = np.dot(v, p)
        if variant in ['p_cg', 'pr_cg', 'mp_cg', 'mpr_cg', 'pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            sigma = np.dot(r, v)
            gamma = np.dot(v, v)
        if variant in ['pr_cg', 'mpr_cg', 'pipe_pr_cg', 'pipe_mpr_cg']:
            alpha = np.dot(r, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            w = w - lamb * u
        last_alpha = alpha
        if variant in ['cg']:
            alpha = np.dot(r, r)
        if variant in ['p_cg', 'pr_cg', 'pipe_p_cg', 'pipe_pr_cg']:
            alpha = alpha - 2*lamb*sigma + lamb**2 * gamma
        if variant in ['mp_cg', 'mpr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            alpha = -alpha + lamb**2 * gamma
        p = r + (alpha / last_alpha) * p
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            v = w + (alpha / last_alpha) * v

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))

        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals


def pcg(A, b, x0, preconditioner=lambda x:x, max_iter=10000, eps=1e-8, variant='cg'):
    """Preconditioned CG method

    Following CG variants are available:
    'cg': vanilla CG method
    'p_cg': Predict-and-Recompute CG method without recomputation
    'pr_cg': Predict-and-Recompute CG method
    'mp_cg': Meurant CG method without recomputation
    'mpr_cg': Meurant CG method
    'pipe_p_cg': Pipelined Predict-and-Recompute CG method without recomputation
    'pipe_pr_cg': Pipelined Predict-and-Recompute CG method
    'pipe_mp_cg': Pipelined Meurant CG method without recomputation
    'pipe_mpr_cg': Pipelined Meurant CG method

    :param A: system matrix A
    :type A: NumPy array
    :param b: right hand side vector b
    :type b: NumPy array
    :param x0: initial guess for the solution
    :type x0: NumPy array
    :param preconditioner: preconditioner, defaults to `lambda x:x`
    :type preconditioner: anonymous function
    :param max_iter: maximum number of iterations, defaults to 10000
    :type max_iter: int, optional
    :param eps: epsilon for stopping criterion, defaults to 1e-8
    :type eps: float, optional
    :param variant: cg method variant, defaults to 'cg'
    :type eps: str, optional
    """    
    r0 = b - A @ x0
    r = r0
    r_tilde = preconditioner(r)
    p = r_tilde
    alpha = np.dot(r_tilde, r)
    if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
        v = A @ p
        v_tilde = preconditioner(v)

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            u = A @ v_tilde
            u_tilde = preconditioner(u)
            w = A @ r_tilde
            w_tilde = preconditioner(w)
        else:
            v = A @ p
            v_tilde = preconditioner(v)
        mu = np.dot(v, p)
        if variant in ['p_cg', 'pr_cg', 'mp_cg', 'mpr_cg', 'pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            sigma = np.dot(r, v_tilde)
            gamma = np.dot(v_tilde, v)
        if variant in ['pr_cg', 'mpr_cg', 'pipe_pr_cg', 'pipe_mpr_cg']:
            alpha = np.dot(r_tilde, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        r_tilde = r_tilde - lamb * v_tilde
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            w = w - lamb * u
            w_tilde = w_tilde - lamb * u_tilde
        last_alpha = alpha
        if variant in ['cg']:
            alpha = np.dot(r_tilde, r)
        if variant in ['p_cg', 'pr_cg', 'pipe_p_cg', 'pipe_pr_cg']:
            alpha = alpha - 2*lamb*sigma + lamb**2 * gamma
        if variant in ['mp_cg', 'mpr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            alpha = -alpha + lamb**2 * gamma
        p = r_tilde + (alpha / last_alpha) * p
        if variant in ['pipe_p_cg', 'pipe_pr_cg', 'pipe_mp_cg', 'pipe_mpr_cg']:
            v = w + (alpha / last_alpha) * v
            v_tilde = w_tilde + (alpha / last_alpha) * v_tilde

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))

        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals
