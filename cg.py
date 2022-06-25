import numpy as np


def cg(A, b, x0, max_iter=10000, eps=1e-8):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    """    
    r0 = b - A @ x0
    p = r0
    alpha0 = np.dot(r0, r0)
    r = r0
    x = x0
    alpha = alpha0
    m = 0
    rm = np.array(np.linalg.norm(b-A@x))
    xm = x
    while (m < max_iter):
        v = A @ p
        mu = np.dot(v, p)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        last_alpha = alpha
        alpha = np.dot(r, r)
        exact_r = b - A @ x
        rm = np.append(rm, np.linalg.norm(exact_r))
        xm = np.vstack((xm, x))
        p = r + (alpha / last_alpha) * p
        m += 1
        if np.linalg.norm(exact_r) < eps:
            break
    return x, rm, xm


def master_cg(A, b, x0, max_iter=10000, eps=1e-8, variant='cg'):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    :param variant: cg method variant ('cg', 'p_cg', 'pr_cg', 'mp_cg', 'mpr_cg')
    """    
    r0 = b - A @ x0
    p = r0
    alpha0 = np.dot(r0, r0)
    r = r0
    x = x0
    alpha = alpha0
    m = 0;
    rm = np.array(np.linalg.norm(b-A@x))
    xm = x
    while (m < max_iter):
        v = A @ p
        mu = np.dot(v, p)
        if variant in ['p_cg', 'pr_cg', 'mp_cg', 'mpr_cg']:
            sigma = np.dot(r, v)
            gamma = np.dot(v, v)
        if variant in ['pr_cg', 'mpr_cg']:
            alpha = np.dot(r, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        last_alpha = alpha
        if variant in ['cg']:
            alpha = np.dot(r, r)
        if variant in ['p_cg', 'pr_cg']:
            alpha = alpha - 2*lamb*sigma + lamb**2 * gamma
        if variant in ['mp_cg', 'mpr_cg']:
            alpha = -alpha + lamb**2 * gamma
        exact_r = b - A @ x
        rm = np.append(rm, np.linalg.norm(exact_r))
        xm = np.vstack((xm, x))
        p = r + (alpha / last_alpha) * p
        m += 1
        if np.linalg.norm(exact_r) < eps:
            break
    return x, rm, xm


def pcg(A, b, x0, x_exact, preconditioner=lambda x:x, max_iter=10000, eps=1e-8, variant='cg'):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    :param variant: cg method variant ('cg', 'p_cg', 'pr_cg', 'mp_cg', 'mpr_cg')
    """    
    r0 = b - A @ x0
    r = r0
    r_tilde = preconditioner(r)
    p = r_tilde
    alpha = np.dot(r_tilde, r)

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        v = A @ p
        v_tilde = preconditioner(v)
        mu = np.dot(v, p)
        if variant in ['cg', 'p_cg', 'pr_cg', 'mp_cg', 'mpr_cg']:
            sigma = np.dot(r, v_tilde)
            gamma = np.dot(v_tilde, v)
        if variant in ['cg', 'pr_cg', 'mpr_cg']:
            alpha = np.dot(r_tilde, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        r_tilde = r_tilde - lamb * v_tilde
        last_alpha = alpha
        if variant in ['cg']:
            alpha = np.dot(r_tilde, r)
        if variant in ['p_cg', 'pr_cg']:
            alpha = alpha - 2*lamb*sigma + lamb**2 * gamma
        if variant in ['mp_cg', 'mpr_cg']:
            alpha = -alpha + lamb**2 * gamma
        p = r_tilde + (alpha / last_alpha) * p

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))

        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals


def pr_pcg(A, b, x0, x_exact, preconditioner=lambda x:x, max_iter=10000, eps=1e-8, recompute=True):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    :param variant: cg method variant ('cg', 'p_cg', 'pr_cg', 'mp_cg', 'mpr_cg')
    """    
    r0 = b - A @ x0
    r = r0
    r_tilde = preconditioner(r)
    p = r_tilde
    alpha = np.dot(r_tilde, r)

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        v = A @ p
        v_tilde = preconditioner(v)
        mu = np.dot(v, p)
        sigma = np.dot(r, v_tilde)
        gamma = np.dot(v_tilde, v)
        if recompute:
            alpha = np.dot(r_tilde, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        r_tilde = r_tilde - lamb * v_tilde
        last_alpha = alpha
        alpha = alpha - 2*lamb*sigma + lamb**2 * gamma
        p = r_tilde + (alpha / last_alpha) * p

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))
        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals


def m_pcg(A, b, x0, x_exact, preconditioner=lambda x:x, max_iter=10000, eps=1e-8, recompute=True):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    :param variant: cg method variant ('cg', 'p_cg', 'pr_cg', 'mp_cg', 'mpr_cg')
    """    
    r0 = b - A @ x0
    r = r0
    r_tilde = preconditioner(r)
    p = r_tilde
    alpha = np.dot(r_tilde, r)

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        v = A @ p
        v_tilde = preconditioner(v)
        mu = np.dot(v, p)
        sigma = np.dot(r, v_tilde)
        gamma = np.dot(v_tilde, v)
        if recompute:
            alpha = np.dot(r_tilde, r)
        lamb = alpha / mu
        x = x + lamb * p
        r = r - lamb * v
        r_tilde = r_tilde - lamb * v_tilde
        last_alpha = alpha
        alpha = -alpha + lamb**2 * gamma
        p = r_tilde + (alpha / last_alpha) * p

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))

        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals


def pipe_pr_pcg(A, b, x0, x_exact, preconditioner=lambda x:x, max_iter=10000, eps=1e-8, recompute=True):
    """_summary_

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param x0: _description_
    :type x0: _type_
    :param max_iter: _description_, defaults to 100
    :type max_iter: int, optional
    :param eps: _description_, defaults to 1e-8
    :type eps: _type_, optional
    """    
    r0 = b - A @ x0
    r = r0
    r_tilde = preconditioner(r)
    p = r_tilde
    alpha = np.dot(r_tilde, r)
    x = x0
    v = A @ p
    v_tilde = preconditioner(v)

    iterates = x0
    residuals = r0
    x = x0
    m = 0
    while (m < max_iter):
        u = A @ v_tilde
        u_tilde = preconditioner(u)
        w = A @ r_tilde
        w_tilde = preconditioner(w)
        mu = np.dot(v, p)
        sigma = np.dot(r_tilde, v)
        gamma = np.dot(v_tilde, v)
        if recompute:
            alpha = np.dot(r_tilde, r)
        lambd = alpha / mu
        x = x + lambd * p
        r = r - lambd * v
        r_tilde = r_tilde - lambd * v_tilde
        w = w - lambd * u
        w_tilde = w_tilde - lambd * u_tilde
        last_alpha = alpha
        alpha = alpha - 2*lambd*sigma + lambd**2 * gamma
        p = r_tilde + (alpha / last_alpha) * p
        v = w + (alpha / last_alpha) * v
        v_tilde = w_tilde + (alpha / last_alpha) * v_tilde

        iterates = np.vstack((iterates, x))
        residuals = np.vstack((residuals, r))

        if np.linalg.norm(b - A @ x) / np.linalg.norm(r0) < eps:
            break
        m += 1
    
    return x, iterates, residuals


def pr_master_cg(A,b,x0,max_iter,variant='', eps=1e-8):
    '''
    master template for predict-and-recompute conjugate gradients
    '''
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    r0 = r_k
    nu_k   =  r_k  @ r_k
    p_k    =  np.copy(r_k)
    s_k    =  A     @ p_k
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    del_k  =  r_k   @ s_k
    gam_k  =  s_k  @ s_k
    w_k    =  A     @ r_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0

    iterates = x_k
    residuals = r_k
    k=0
    # run main optimization
    for k in range(1,max_iter):
        print(x_k[:4])
        # update indexing
        a_k2   =  a_k1
        a_k1   =  a_k
        b_k1   =  b_k
        nu_k1  =  nu_k
        del_k1 =  del_k
        gam_k1 =  gam_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        w_k1   =  np.copy(w_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        w_k    =  A     @ r_k
        nu_k   = - nu_k1 + a_k1**2 * gam_k1 if variant == 'm' else \
                 nu_k1 - 2 * a_k1 * del_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  r_k   +    b_k * p_k1
        s_k    =  A     @ p_k
        mu_k   =  p_k   @ s_k 
        del_k  =  r_k   @ s_k
        gam_k  =  s_k   @ s_k
        nu_k   =  r_k   @ r_k
        a_k    =  nu_k / mu_k

        iterates = np.vstack((iterates, x_k))
        residuals = np.vstack((residuals, r_k))

        if np.linalg.norm(b - A @ x_k) / np.linalg.norm(r0) < eps:
            break
            
    return x_k, iterates, residuals

def pr_master_pcg(A,b,x0,max_iter,preconditioner=lambda x:x,variant='', eps=1e-8):
    '''
    master template for predict-and-recompute conjugate gradients (preconditioned)
    '''
    
    x_k    =  np.copy(x0)
    r_k    =  np.copy(b - A @ x_k)
    r0 = r_k
    rt_k   =  preconditioner(r_k)
    nu_k   =  rt_k  @ r_k
    p_k    =  np.copy(rt_k)
    s_k    =  A     @ p_k
    st_k   =  preconditioner(s_k)
    mu_k   =  p_k   @ s_k
    a_k    =  nu_k / mu_k
    del_k  =  r_k   @ st_k
    gam_k  =  st_k  @ s_k
    a_k1   =  0
    a_k2   =  0
    b_k    =  0
    b_k1   =  0

    iterates = x_k
    residuals = r_k
    k=0
    # run main optimization
    for k in range(1,max_iter):

        # update indexing
        a_k2   =  a_k1
        a_k1   =  a_k
        b_k1   =  b_k
        nu_k1  =  nu_k
        del_k1 =  del_k
        gam_k1 =  gam_k
        
        x_k1   =  np.copy(x_k)
        r_k1   =  np.copy(r_k)
        rt_k1  =  np.copy(rt_k)
        p_k1   =  np.copy(p_k)
        s_k1   =  np.copy(s_k)
        st_k1  =  np.copy(st_k)
        
        # main loop
        x_k    =  x_k1  +   a_k1 * p_k1
        r_k    =  r_k1  -   a_k1 * s_k1
        rt_k   =  rt_k1 -   a_k1 * st_k1
        nu_k   = - nu_k1 + a_k1**2 * gam_k1 if variant == 'm' else nu_k1 - 2 * a_k1 * del_k1 + a_k1**2 * gam_k1
        b_k    =  nu_k / nu_k1
        p_k    =  rt_k  +   b_k * p_k1
        s_k    =  A     @ p_k
        st_k   =  preconditioner(s_k)
        mu_k   =  p_k   @ s_k 
        del_k  =  r_k   @ st_k
        gam_k  =  st_k  @ s_k
        nu_k   =  rt_k  @ r_k 
        a_k    =  nu_k / mu_k
        
        exact_r = b - A @ x_k
        iterates = np.vstack((iterates, x_k))
        residuals = np.vstack((residuals, r_k))
        if np.linalg.norm(b - A @ x_k) / np.linalg.norm(r0) < eps:
            break

    return x_k, iterates, residuals
