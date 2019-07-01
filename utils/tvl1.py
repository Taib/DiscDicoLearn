# -*- coding: utf-8 -*-
"""Total variation methods"""
 
# Author: http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html


import numpy as np

def nabla(I):
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G

def nablaT(G):
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I
# little auxiliary routine
def anorm(x):
    '''Calculate L2 norm over the last array dimention'''
    return np.sqrt((x*x).sum(-1))

def calc_energy_ROF(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = 0.5 * clambda * ((X - observation)**2).sum()
    return Ereg + Edata

def calc_energy_TVL1(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = clambda * np.abs(X - observation).sum()
    return Ereg + Edata
def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P)/r)
    return P / nP[...,np.newaxis]
    
def shrink_1d(X, F, step):
    '''pixel-wise scalar srinking'''
    return X + np.clip(F - X, -step, step)


def solve_ROF(img, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    for i in xrange(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        lt = clambda * tau
        X1 = (X - tau * nablaT(P) + lt * img) / (1.0 + lt)
        X = X1 + theta * (X1 - X)
        if i % 10 == 0:
            print "%.2f" % calc_energy_ROF(X, img, clambda),
    print
    return X
    
def solve_TVL1(img, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    E = []
    for i in xrange(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        X1 = shrink_1d(X - tau*nablaT(P), img, clambda*tau)
        X = X1 + theta * (X1 - X) 
        E.append(calc_energy_TVL1(X, img, clambda)) 
    return X, E 

def solve_TVL1_multy(imgs, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    imgs = np.array(imgs)
    X = imgs[0].copy()
    P = nabla(X)
    Rs = np.zeros_like(imgs)
    for i in xrange(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        Rs = np.clip(Rs + sigma*(X-imgs), -clambda, clambda)
        X1 = X - tau*(nablaT(P) + Rs.sum(0))
        X = X1 + theta * (X1 - X)
    return X


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from scipy.misc import lena
    img_ref = lena()[140:,120:][:256,:256] / 255.0
    
    def make_noisy(img):
        # add gaussian noise
        img = img + 0.1 * np.random.normal(size=img.shape)
        # add some outliers in on the right side of the image
        m = np.random.rand(*img.shape) < 0.2
        m[:,:160] = 0
        img[m] = np.random.rand(m.sum())
        return img
    def make_spotty(img, r=3, n=1000):
        img = img.copy()
        h, w = img.shape
        for i in xrange(n):
            x, y = np.int32(np.random.rand(2)*(w-r, h-r))
            img[y:y+r, x:x+r] = round(np.random.rand())
        return img

    img_obs = make_noisy(img_ref)
    plt.figure();plt.imshow(img_ref, cmap=plt.cm.gray)
    plt.figure();plt.imshow(img_obs, cmap=plt.cm.gray)
    plt.figure();plt.imshow(solve_TVL1(img_obs, 1.0)[0], cmap=plt.cm.gray) 
        
    observations = [make_spotty(make_noisy(img_ref)) for i in xrange(5)]
    plt.figure();plt.imshow(observations[4], cmap=plt.cm.gray)
    plt.figure();plt.imshow(solve_TVL1_multy(observations, 0.5), cmap=plt.cm.gray)
