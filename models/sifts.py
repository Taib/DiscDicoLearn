# -*- coding: utf-8 -*-
"""Scale Invariant Feature Transform"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@etu.upmc.fr> 
 
import numpy as np
from scipy import signal 

def gauss_map(s, sigma=None):
    sigma = 0.5 * s if sigma is None else sigma
    N2    = np.ceil(s/2.) 
    Mg    = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            Mg[i,j] = np.power(2*np.pi*sigma**2, -1)*np.exp(-((i-N2)**2 + (j-N2)**2)/(2*sigma**2))
    return Mg

def orientation(Ix, Iy, Ig, bins=8):
    ori = np.zeros((bins, 2))
    for i in range(bins):
        ori[i, 0] = np.cos(2*np.pi*(i-1)/bins)
        ori[i, 1] = np.sin(2*np.pi*(i-1)/bins)
    
    Ior = np.zeros_like(Ig)
    for i in range(Ior.shape[0]):
        for j in range(Ior.shape[1]):
            if (Ig[i,j] > 0):
                v = np.array([Ix[i,j], -Iy[i,j]])
                v = v/np.linalg.norm(v)
                p = np.dot(ori, v)
                Ior[i,j] = np.argmax(p)
    return Ior.astype('int')
                
def compute_sift(s, Ig, Ior, Mg, bins=8):
    sift = np.zeros((s, bins))
    z    = 0
    for i in range(0,s,4):
        for j in range(0,s,4):
            H = np.zeros(bins)
            for l in range(i,i+3):
                for k in range(j,j+3):
                    if Ior[l,k] >0:
                        H[Ior[l,k]] = H[Ior[l,k]] + Ig[l,k]*Mg[l,k]
                       
            sift[z,:] = H
            z = z+1
            
    sift = sift.flatten()  
    if np.linalg.norm(sift) <= 1e-4:
        sift = np.zeros(s*bins) 
    return sift

def sift(im, s=None, bins=8):
    s    = s if s is not None else im.shape[0]
    
    kern = np.outer([-1, 0, 1], [1, 2, 1])   
    Ix   = signal.convolve2d(im, kern, mode='same', boundary='symm')
    Iy   = signal.convolve2d(im, kern.T, mode='same', boundary='symm')  
    Ig   = np.sqrt(Ix**2 + Iy**2)
    Ior  = orientation(Ix, Iy, Ig)
    Mg   = gauss_map(s)
    sift = compute_sift(s, Ig, Ior, Mg)
    
    return sift
