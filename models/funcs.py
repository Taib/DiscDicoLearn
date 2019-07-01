# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:26:37 2017

@author: tramx
"""
 
import numpy as np 
from sifts import sift, gauss_map
from joblib import Parallel, delayed
from skimage import feature, filters


from sklearn.ensemble import RandomForestClassifier   
def classifier(X, y, name='rf', n_jobs=8):
    if name=='rf':
        model = RandomForestClassifier(100, n_jobs=n_jobs, verbose=10)
        model = model.fit(X, y) 
    return model

def softmax(a, axis=0):
    e = np.exp(a)
    return e/e.sum(axis) 
    
def zca_whitening(x):
    assert(x.ndim == 2)
    s = np.dot(x, x.T)/x.shape[1]
    u,d,v = np.linalg.svd(s)
    w = np.dot(np.dot(u, np.diag(1.0/np.sqrt(d + 0.1))), u.T)
    xh = np.dot(w, x.T).T
    return xh, w 

def whitening(x):
    assert(x.ndim == 2)
    s = np.dot(x, x.T)/x.shape[1]
    u,d,v = np.linalg.svd(s)
    w = np.dot(np.dot(u, np.diag(1.0/np.sqrt(d + 0.01))), u.T)
    xh = np.dot(w, x.T).T
    return xh, w  

def dice(s,g):
    o = np.unique(s)
    
    s = s.flatten()
    g = g.flatten() 

    dice = []
    for k in o:
        if k == 0:
            continue 
        ss = set(list(np.squeeze(np.argwhere(s==k))))
        gs = set(list(np.squeeze(np.argwhere(g==k))))
        if (len(ss) + len(gs)) != 0:
            dice.append(len(ss.intersection(gs))*2.0 / (len(ss) + len(gs)))
        else:
            dice.append(1)
        
    return np.mean(dice)


def img_glcm(img, args={}):
    assert(img.ndim == 2)    
    ang = args['angles'] if args.has_key('angles') else [0, np.pi/4, np.pi/2, 3*np.pi/4]
    dis = args['distance'] if args.has_key('distance') else [1]
    nor = args['normed'] if args.has_key('normed') else True
    
    g  = feature.greycomatrix(img, dis, ang, normed=nor)
    pr = [feature.greycoprops(g, 'contrast'), feature.greycoprops(g, 'homogeneity'),
          feature.greycoprops(g, 'dissimilarity'), feature.greycoprops(g, 'correlation') ]
    return np.array(pr).flatten()
 
def patches_glcm(patches, p_shape=None, args={}):
    l       = int(np.sqrt(patches.shape[1])) 
    p_shape = (l, l) if p_shape is None else p_shape
    
    props = Parallel(n_jobs=8)(delayed(img_glcm)(x.reshape(p_shape))
                                for x in patches) 
    return np.array(props)
 

def patches_gaussian(patches, p_shape=None, sigma=1.):
    l       = patches.shape[0]
    p_shape = (int(np.sqrt(l)), int(np.sqrt(l))) if p_shape is None else p_shape
    patches = patches
    patches = [filters.gaussian(patches[i].reshape(p_shape), sigma).flatten() for i in xrange(patches.shape[0])]

def patches_sift(patches, p_shape=None, sigma=1.):
    l       = int(np.sqrt(patches.shape[1])) 
    p_shape = (l, l) if p_shape is None else p_shape
      
    sifts   = Parallel(n_jobs=8, verbose=10)(delayed(sift)(x.reshape(p_shape))
                                for x in patches)
    
    return np.array(sifts)
    
def norm_cols(X, soft_norm=True): 
    """Normalize each row of X to have a l_2-norm >= 1
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Patch vectors, where n_samples is the number of samples and
        n_features is the number of features.
    soft_eps : bool, Optional 
             if True the vectors are projected onto the closed unit-ball
             else each vectors must have a l_2-norm of 1 of 0
             
    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Patch matrix  with normalized rows.
        
    """    
    norms = np.sqrt((X ** 2).sum(axis=1))
    ids   = norms != 0
    if soft_norm:
        ids *= (norms > 1) 
    X[ids, :] /=np.tile(norms[ids],(X[ids, :].shape[1],1)).T
    return X
    
def normalize_rgb_patches(X, normal_m, **kwargs):
    # similar normalization as for gray patches, except it's performed by channel    
    if normal_m == 0: 
        X = X/255. if X.max() > 1 else X
        psize = X.shape[1]/3
        for c in range(3):
            X[:, c*psize:c*psize+psize] = norm_cols(X[:, c*psize:c*psize+psize])
        return X 
        
    print '[normalization] mode %d does not exist.' %normal_m
    return X

def normalize_gray_patches(X, normal_m, **kwargs):  
    """Normalize each row of X using a specific technique defined by "normal_m"
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Patch vectors, where n_samples is the number of samples and
        n_features is the number of features.
    normal_m : integer in {0, ..., 10}
        normalization technique
    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Patch matrix  with normalized rows.
        
    """    
    if normal_m == 0: 
        X = X/255. if X.max() > 1 else X
        X = norm_cols(X)
        return X
        
    if normal_m == 1: 
        X = X/255. if X.max() > 1 else X
        X -= np.tile(X.mean(axis=0), (X.shape[0], 1))
        return X
    
    if normal_m == 2: 
        X = X/255. if X.max() > 1 else X
        X -= np.tile(X.mean(axis=1), (X.shape[1], 1))
        X = norm_cols(X)
        return X
    
    if normal_m == 3:
        mean_patch = kwargs['mean_patch'] if kwargs.has_key('mean_patch') else None
        assert(mean_patch is not None ) 
        X -= np.tile(255.-mean_patch, (X.shape[0], 1)).T
        X = norm_cols(X)
        return X
    
    if normal_m == 4: 
        p_shape = kwargs['p_shape'] if kwargs.has_key('p_shape') else None
        patches = patches_glcm(X, p_shape)
        patches = norm_cols(patches)
        return patches
     
    if normal_m == 6:
        X = X/255. if X.max() > 1 else X
    
    if normal_m == 7:
        X = X/255. if X.max() > 1 else X
        p_shape = kwargs['p_shape'] if kwargs.has_key('p_shape') else None
        sigma   = kwargs['gaussian_sigma'] if kwargs.has_key('gaussian_sigma') else .5
        X       = patches_gaussian(X, p_shape, sigma)
        return X
    
    if normal_m == 8:
        X = X/255. if X.max() > 1 else X
        p_shape = kwargs['p_shape'] if kwargs.has_key('p_shape') else None
        sigma   = kwargs['gaussian_sigma'] if kwargs.has_key('gaussian_sigma') else .5
        X       = patches_sift(X, p_shape, sigma)
        return X
        
    if normal_m == 9:
        X = X/255. if X.max() > 1 else X
        p_shape = kwargs['p_shape'] if kwargs.has_key('p_shape') else None
        sigma   = kwargs['gaussian_sigma'] if kwargs.has_key('gaussian_sigma') else .5
        sifts   = patches_sift(X, p_shape, sigma)
        X       = np.vstack((X, sifts))
        return X
           
    if normal_m == 10:
        X = X/255. if X.max() > 1 else X
        p_shape = kwargs['p_shape'] if kwargs.has_key('p_shape') else None
        sigma   = kwargs['sigma_patches'] if kwargs.has_key('sigma_patches') else 4
        size    = int(np.sqrt(patches.shape[0])) if p_shape is None else p_shape[0]
        gauss   = gauss_map(size, sigma).flatten()
        gauss   = np.tile(gauss, (X.shape[1], 1)).T
        
        #X       = norm_cols(X)
        X       = X*gauss
        return X
    
    return X

def normalize_patches(X, is_color, normal_m, **kwargs):
    if is_color:
        return normalize_rgb_patches(X, normal_m, **kwargs)
    return normalize_gray_patches(X, normal_m, **kwargs)