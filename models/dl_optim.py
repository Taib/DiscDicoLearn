# -*- coding: utf-8 -*-
"""This module implements some sparse coding and dictionary learning algorithms:
    - the online dictionary learning (using spams), 
    - standard sparse conding: lasso, basis pursuit, etc. (using spams), 
    - the K-SVD,
    - the graph sparse coding, 
    - the graph embedded dictionary learning,
    - the kernel dictionary learning, 
    - the kernel OMP,
    - the Analysis-KSVD,
    - the Backward-Greedy,
"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 

import numpy as np  
import spams 
from scipy import linalg, sparse
from joblib import Parallel, delayed

def _compute_codes(X, D, sc_mode, W, lambda1, lambda2, 
                         n_jobs, pos_coef, **args):
    """ Deprecated for very-large datasets! Use sparse_decode instead.
    """                               
    X = np.asfortranarray(X)
    D = np.asfortranarray(D)
     
    gram = None
    cov = None
    if W is None:
        gram = np.dot(D.T, D) 
        gram = np.asfortranarray(gram)
        cov  = np.dot(D.T, X)  
        cov  = np.asfortranarray(cov)
    
    
    if sc_mode in [0, 1, 2]:
        if W is None:
            A = spams.lasso(X , D, gram, cov, lambda1=lambda1, lambda2=lambda2,
                            numThreads=n_jobs, mode=sc_mode, pos=pos_coef)
        else:
            A = spams.lassoWeighted(X , D, W, lambda1=lambda1,mode=sc_mode, 
                                    pos=pos_coef, numThreads=n_jobs)
            
    else:
        L        = lambda1 if sc_mode == 3 else None
        eps      = lambda1 if sc_mode == 4 else None
        lambda_1 = lambda1 if sc_mode == 5 else None
        A = spams.omp(X, D, L, eps, lambda_1, numThreads=n_jobs)
        
    return A.toarray()

def sparse_decode(X, D, sc_mode=2, W=None, lambda1=None, lambda2=0., n_jobs=8, pos_coef=False, **args):    
    assert(X.shape[0] == D.shape[0])  
    
    if X.shape[1] > 500000: 
        print('[sparse decode]', X.shape, D.shape)
        data = np.array_split(X, 100, axis=1)
        A = Parallel(n_jobs=n_jobs)(delayed(_compute_codes)(d, D, sc_mode,
                                                          W, lambda1, lambda2, 
                                                          n_jobs, pos_coef, **args)
                                   for d in data)
        A = np.asfortranarray(np.concatenate(A, axis=1), dtype=D.dtype) 
        return A
    else:
        return _compute_codes(X, D, sc_mode, W, lambda1, lambda2, 
                             n_jobs, pos_coef, **args)
                             

def odl(X, n_atoms=100, lambda1=10, mode=3, modeD=0,
        n_iter=1000, n_jobs=8, batchsize=512, init_D=None, 
        model=None, return_model=True, lambda2=0.,
        pos_coef=False, pos_dico=False, **args):
    if init_D is None:
        D = np.array(np.zeros((X.shape[0], n_atoms)),dtype=X.dtype,order='FORTRAN') 
    else:
        D=np.asfortranarray(init_D)
    
    if model is None:      
        model           = {}
        model['A']      = np.zeros((D.shape[1], X.shape[0]),dtype=X.dtype,order="FORTRAN")
        model['B']      = np.zeros((D.shape[0], D.shape[1]),dtype=X.dtype,order="FORTRAN")
        model['iter']   = 0
    
    param = {'K': n_atoms, 'lambda1': lambda1 , 'numThreads': n_jobs,
             'batchsize': batchsize, 'iter': n_iter, 'mode': mode,
             'modeD':  modeD, 'lambda2': lambda2, 'D': D, 'model': model,
             'return_model': return_model, 'posAlpha': pos_coef, 
             'posD': pos_dico, 'whiten':False, 'verbose':10}   
             
    data = np.asfortranarray(X, dtype=X.dtype) 

    return spams.trainDL(data,return_model= return_model,model= model,D = D,
                        numThreads = n_jobs,batchsize = batchsize,K= n_atoms, lambda1=lambda1,
                        lambda2= lambda2, iter=n_iter, mode=mode,  modeD=modeD)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



def ksvd_optim_dk(X, D, A, k):
    
    wk = np.argwhere(A[k,:] != 0).squeeze()
    
    # To handle non-used atoms
    if wk.size == 0:
        rcol = np.random.randint(0, X.shape[1])
        return [X[:, rcol], 1, rcol]  
    
    Ek = X - np.dot(D, A) + np.outer(D[:, k], A[k, :])
    Ek = Ek[:, wk]
    Ek = Ek if Ek.ndim > 1 else Ek.reshape((-1,1))
    
    U, S, V  = np.linalg.svd(Ek, False)
    
    return [U[:, 0], S[0]*V[0,:], wk]



def ksvd(X, lambda1=None, model=None, return_model=False, n_atoms=100,
         n_iter=1000, n_jobs=8):
    if (model is None):
        D = np.random.randn(X.shape[1], n_atoms)
        D = spams.normalize(np.asfortranarray(D, dtype=D.dtype))
        
        A = np.random.randn(n_atoms, X.shape[0])
    else:
        D = model[0].T
        A = model[1].T
    
         
    E = np.zeros(n_iter)
    for i in range(n_iter): 
        print i
        # Update code
        A = spams.omp(X.T, D, L=lambda1, numThreads=n_jobs)  
        # Update Dico  --- should be parallelized
        for k in np.arange(n_atoms):
            print k, A.shape, D.shape, np.dot(D, A).shape, X.T.shape
            res = ksvd_optim_dk(X.T, D, A, k)
            if res is not None:
                D[:, k]  = res[0]             
                A[k, res[2]] = res[1] 
        E[i] = ((X.T - np.dot(D, A))**2).sum()
        print 'ksvd iter', i, ' --> ', E[i]
    
    if return_model:
        return D.T, [A.T, D.T]
    
    return D.T, None


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


def graph_sc(X, D, Lc, beta=0.6, rho=0.01, T=None, sc_n_iter=10, **args):

    # Update code
    A = sparse_decode(X, D.T, sc_mode=3, lambda1=T).T  
    
    Z = A.copy()
    U = 0
    
    i = 0
    Lm1 = np.dot(D.T, D) + rho*np.identity(D.shape[1])
    Lm2 = beta*Lc.toarray() if  sparse.issparse(Lc) else beta*Lc
    P3  = np.dot(D.T, X)
    while i < sc_n_iter:
        i += 1
        Rm = P3 + rho*(Z - U)  
        A  = linalg.solve_sylvester(Lm1, Lm2, Rm)
        
        Z   = A + U 
        ids = np.argsort(np.abs(Z), 0)[::-1]
        ids = ids[T:, :].T
        
        for k in range(Z.shape[1]):
            Z[ids[k], k] = 0
            
        U = U + A - Z
        
        for k in range(Z.shape[1]):
            wk       = np.argwhere(Z[:,k] != 0).squeeze()
            Z[wk, k] = np.dot(linalg.pinv(D[:,wk]), X[:,k])
            
    return Z


def graph_optim_dk(X, D, A, k, beta, Lc, L, alpha):
    
    wk = np.argwhere(A[k,:] != 0).squeeze()
    
    # To handle non-used atoms
    if wk.size == 0: 
        return None
    
    others = np.arange(D.shape[1])
    others = np.delete(others, k)
    
    Ek = np.dot(D[:, others], A[others, :])
    Ek = X - Ek
    Ek = Ek[:, wk]
    Ek = Ek if Ek.ndim > 1 else Ek[..., np.newaxis]
    
    Lcwk = Lc.toarray()[wk,wk] if  sparse.issparse(Lc) else Lc[wk,wk]
    inv1 = np.linalg.inv((linalg.norm(D[:, k])**2)*np.identity(wk.size) + beta*Lcwk) 
    ar   = np.dot(inv1, np.dot(Ek.T, D[:,k]))
    
    inv2 = np.linalg.inv((linalg.norm(ar)**2)*np.identity(L.shape[1]) + alpha*L)
    dk   = np.dot(inv2, np.dot(Ek, ar)) 
    
    return [dk, ar, wk]



def manifold_laplacian(label):
    Wei = 0
    for l in np.unique(label): 
        Wei += np.outer(label == l, label == l)  
    Deg = np.diag(Wei.sum(0)) 
    Lc  = sparse.lil_matrix(Deg - Wei)
    return Lc

def graphDL(X, Lc, L=None, lambda1=None, model=None, n_atoms=100, n_iter=1000,
             beta=0.3, rho=0.01, alpha=0.1, sc_n_iter=100, return_model=True):
    
    if (model is None):
        D = np.random.randn(X.shape[1], n_atoms)
        D = D/np.tile(np.sqrt((D ** 2).sum(axis=0)), (D.shape[0],1))
        
        A = np.random.randn(n_atoms, X.shape[0])
    else:
        D = model[0].T
        A = model[1].T
            
    L = np.zeros((X.shape[1], X.shape[1])) if L is None else L
     
    E = np.zeros(n_iter)
     
   
    for i in range(n_iter): 
        
        #print '--- Update code'
        A = graph_sc(X.T, D, Lc, beta, rho, lambda1, sc_n_iter)  
        
        #print '--- Update Dico'
        for k in np.arange(D.shape[1]):
            res = graph_optim_dk(X, D, A, k, beta, Lc, L, alpha)
            if res is not None:
                D[:, k]  = res[0]             
                A[k, res[2]] = res[1] 
        
        E[i] = ((X.T - np.dot(D, A))**2).sum()
        print 'graphDL iter', i, ' --> ', E[i]
  
    if return_model:
        return D.T, [A.T, D.T]
    
    return D.T, None



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


def komp(p, B, K, T, **args):
    """
        K = K(Y, Y) positive semi-definite
        p = K(z, Y)
    """   
    assert(K.ndim == 2 and K.shape[1] == K.shape[0]) 
     
    a = 0
    v = np.zeros(K.shape[0])
     
    Is = np.zeros(B.shape[1], dtype='int')
    for s in range(T): 
        tau = np.squeeze(np.dot(p - np.dot(v, K), B[:, (1-Is).astype('bool')]))  
        i   = np.argmax(np.abs(tau))  
        Is[np.squeeze(np.nonzero(1-Is))[i]] = 1
        
        R   = B[:, Is.astype('bool')]
        
        inv = linalg.inv(np.dot(R.T, np.dot(K, R)))
        a   = np.dot(inv, np.dot(p, R).T)
        v   = np.dot(R, a).T
    
    res = np.zeros(B.shape[1])
    res[Is.astype('bool')] = a 
    
    return res


def kernel_optim_dk(K, B, A, k):
    
    wk = np.argwhere(A[k,:] != 0).squeeze()
    
    # To handle non-used atoms
    if wk.size == 0: 
        return None 
    
    
    others = np.arange(B.shape[1])
    others = np.delete(others, k)
    
    Ek = np.dot(B[:, others], A[others, :])
    Ek = np.identity(Ek.shape[0]) - Ek
    Ek = Ek[:, wk]
    
    Ek = Ek if Ek.ndim > 1 else Ek[...,np.newaxis]
    
    U, S, V  = np.linalg.svd(np.dot(Ek.T, np.dot(K, Ek)), full_matrices=0)
    
    return [np.power(S[0], 0.5)*np.dot(Ek, U[:, 0]), np.power(S[0], 0.5)*V[0,:], wk]


def kernel_project(z, X, sigma=1., const=1., deg=4, kern='gauss'):
    assert(z.ndim == 1 and X.ndim == 2) 
    
    if kern == 'poly':
        p = np.dot(z[np.newaxis,...], X.T)
        p += const
        p = np.power(p, deg)
        return p
    if kern == 'gauss':
        p = X.T - np.tile(z, (X.shape[0],1)).T
        p = np.exp(- (p**2).sum(0)/(2*sigma**2))
        return p


def kernel_matrix(X, sigma=.9, const=1., deg=4, kern='gauss'): 
    K = []
    append = K.append
    for x in X:  
        append(kernel_project(x, X, sigma, const, deg, kern)) 
    K = np.array(K)
    return K.squeeze()

def kernelDL(K, lambda1, model=None, return_model=True,
             n_atoms=100, n_iter=100, n_jobs=8):

    if (model is None): 
        D = np.random.randn(K.shape[1], n_atoms)
        D = D/np.tile(np.sqrt((D ** 2).sum(axis=0)), (D.shape[0],1))
     
        A = np.random.randn(n_atoms, K.shape[1])
    else:
        D = model[0].T
        A = model[1].T
    
    E = np.zeros(n_iter)
     
    
    I = np.identity(K.shape[0])
    for i in range(n_iter): 
         
        A = []
        append = A.append
        for k in range(K.shape[1]):
            p = K[k,:] 
            append(list(komp(p, D, K, lambda1)))
        A = np.array(A).T
        
        #print '--- Update Dico'
        for k in np.arange(D.shape[1]):
            res = kernel_optim_dk(K, D, A, k)
            if res is not None:
                D[:, k]  = res[0]             
                A[k, res[2]] = res[1]
         
         
        aux = (I - np.dot(D, A))
        E[i] = np.trace(np.dot(aux.T, np.dot(K, aux)))
        print 'kernelDL iter', i, ' --> ', E[i]
            
    if return_model:
        return D.T, [A.T, D.T]
    return D.T, None


