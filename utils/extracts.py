# -*- coding: utf-8 -*-
"""Usefull methods for patch extraction"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 
 
import numpy as np  
from skimage import util, segmentation, color
 
def extract_patches(img, p_shape, nb_patches=None, given_ids=None, anno_img=None,
                    anno_map=False, return_ids=False, mask=None):
    """Extract patches from the image img
    
    Random extraction if "given_ids" is None.
    Patches of all the pixels are extracted if "given_ids" and "nb_patches" are 
    both None.
    
    Parameters
    ----------
    img : ndarray, shape (height, width, n_channel),
        The input image (n_channel = 3 for RGB images)
    p_shape : tuple, 
             the patches shape (height, width)
    nb_patches : long, Optional
            number of patch to extract
    given_ids : array, Optional, shape (nb_patches, )
        Indices of the pixels on which the patches are to be extracted. 
        Given an image of shape (h,w), the index of the pixel (i, j) is: i*w+j
    anno_img : ndarray, Optional,  shape (height, width)
        If given the labels of each patch are also extracted
    anno_map : bool or tuple, Default False
        if True a classification map is assotiated with each patch 
        instead of one single class label. if a tuple is given, the classification
        map will be of shape the given tuple instead of "p_shape"
    return_ids : bool, Default False
        if "given_ids" is to be returned. Usefull, when willing to extract 
        patches in the same positions from different images of same shape.
    mask : ndarray, shape (height, width), Optional
        a boolean image of the same shape as the input "img". If given, only
        the pixel with true value in mask will be considered.
    
    Returns
    -------
    Patches : array-like, shape (n_samples, n_features)
        Patche vectors, where n_samples is the number of samples and
        n_features is the number of features.
    Labels : if "anno_img" is given, array-like,  
            shape (n_samples,) or (n_samples, n_features) if anno_map is True
    given_ids : if "return_ids" is given, array-like,  
            shape (n_samples,) 
        
    """     
    assert(img.ndim >= 2)
    
    all_pix = False
    if nb_patches is  None:
        nb_patches = np.prod(img.shape[:2]) if given_ids is None else len(given_ids)
        all_pix    = True

    # add one pixel in padding to avoid extreme cases
    #for odd patch shapes
    adl = 0 if p_shape[0]%2 == 0 else 1
    adc = 0 if p_shape[1]%2 == 0 else 1
    
    if given_ids is None:
        maskvals = mask.flatten() if mask is not None else None
        if anno_img is not None:
            if not all_pix:                
                given_ids = [] 
                cls       = np.unique(anno_img)
                per_cls   = nb_patches/len(cls)
                for i in cls: 
                    search_area = anno_img.flatten()==i
                    if maskvals is not None: 
                        search_area = search_area*maskvals
                    cl_sample = np.squeeze(np.nonzero(search_area)) 
                    np.random.shuffle(cl_sample)  
                    rids = np.random.randint(0, cl_sample.size, per_cls) ###
                    cl_sample = cl_sample[rids] 
                    given_ids.append(list(cl_sample))
                    
                given_ids = np.array(given_ids).flatten()  
            else:
                given_ids = np.arange(nb_patches) 
        else:
            if not all_pix:
                if maskvals is not None:
                    maskvals = np.squeeze(np.nonzero(maskvals))
                    np.random.shuffle(maskvals) 
                    given_ids = maskvals[:nb_patches] 
                else:
                    rrows = np.random.randint(0, img.shape[0], nb_patches)  
                    given_ids = rrows 
            else:
                given_ids = np.arange(nb_patches)     
        #to have exactly nb_patches
        if(given_ids.size < nb_patches):
            sup = np.random.randint(0, anno_img.size, nb_patches - given_ids.size)
            given_ids = np.append(given_ids, sup) 
        
    img     = list(np.reshape(img, (img.shape[0], img.shape[1], -1)).transpose(2,0,1))
    for i in range(len(img)):
        img[i] = util.pad(img[i], (p_shape[0]/2+adl, p_shape[1]/2+adc), mode='reflect') 
    img = np.array(img).transpose(1,2,0)
    
    patches = np.zeros((len(given_ids), np.prod(p_shape)*img.shape[2]))
    anno_map= p_shape if anno_map and anno_map == True else anno_map
    if anno_img is not None:
        if not anno_map: 
            labels   = np.zeros((len(given_ids)))  
        else:
            al = 0 if anno_map[0]%2 == 0 else 1
            ac = 0 if anno_map[1]%2 == 0 else 1
            labels   = np.zeros((len(given_ids), np.prod(anno_map))) 
            anno_img = util.pad(anno_img, (anno_map[0]/2+al, anno_map[1]/2+ac),
                                mode='reflect')
 
    s = np.prod(p_shape)
    for k in range(len(given_ids)): 
        i = given_ids[k]/(img.shape[1] - p_shape[0] - adl) 
        j = given_ids[k]%(img.shape[1] - p_shape[1] - adc) 
         
        # to concatenate the bitmaps from each channel instead of mixing them 
        # with a simple flatten
        for c in range(img.shape[2]): 
            patches[k, c*s:c*s+s] = img[i:i+p_shape[0], j:j+p_shape[1], c].flatten()
        
        if anno_img is not None and not anno_map:
            labels[k] = anno_img[i, j]
        elif anno_img is not None: 
            labels[k, :] = anno_img[i:i+anno_map[0], j:j+anno_map[1]].flatten()
     
    if anno_img is not None and return_ids :
        return patches, labels, given_ids
    elif return_ids:
        return patches, given_ids
    elif anno_img is not None:
        return patches, labels
    return patches

def image_patches_merging(patches, im_shape, p_shape, given_ids=None,
                          return_cnt=False):
    """Merge patches to reconstruct an image
     
    
    Parameters
    ----------
    Patches : array-like, shape (n_samples, n_features)
        Patche vectors, where n_samples is the number of samples and
        n_features is the number of features.
    im_shape : tuple, 
             the shape of output image (height, width)
    p_shape : tuple, 
             the patches shape (height, width) 
    given_ids : array-like, Optional, shape (n_samples)
             The pixel index of each patch. Usefull when the patches are taken
             using a stride. 
    
    Returns
    -------
    img : ndarray, shape (height, width,),
        The reconstructed image.
    cnt : if "return_cnt" is True, ndarray, shape (height, width),
        A matrix that indicates the number of time a pixel is used.
        
    """    
    if len(im_shape) == 2:
        rgb = False
        assert(patches.shape[1] == np.prod(p_shape)) 
    else:
        rgb = True
        # only RGB-like image - the last dimension is considered as the channel one.
        assert(len(im_shape) == 3 and im_shape[-1] == 3) 
    
    # remove the additional one pixel for odd patch shapes
    adl = 0 if p_shape[0]%2 == 0 else 1
    adc = 0 if p_shape[1]%2 == 0 else 1
     
    if not rgb:
        img = np.zeros((im_shape[0]+p_shape[0], im_shape[1]+p_shape[1]))   
    else:
        img = np.zeros((im_shape[0]+p_shape[0], im_shape[1]+p_shape[1], 3))    
    cnt = np.zeros((im_shape[0]+p_shape[0], im_shape[1]+p_shape[1]))

    s = np.prod(p_shape)
    given_ids = np.arange(patches.shape[0]) if given_ids is None else given_ids
    for p in range(len(given_ids)):
        i = given_ids[p]/(im_shape[1]) 
        j = given_ids[p]%(im_shape[1]) 
        if not rgb: 
            img[i:i+p_shape[0], j:j+p_shape[1]] += patches[p, :].reshape(p_shape) 
        else:  
            for c in range(3):
                img[i:i+p_shape[0], j:j+p_shape[1], c] += patches[p, c*s:c*s+s].reshape(p_shape)
        cnt[i:i+p_shape[0], j:j+p_shape[1]] += 1
        
    """
    s = np.prod(p_shape)
    for i in range(img.shape[0] - p_shape[0]):
        for j in range(img.shape[1] - p_shape[1]):  
            pos = i*(img.shape[1] - p_shape[1] ) + j 
            if not rgb:
                img[i:i+p_shape[0], j:j+p_shape[1]] += patches[pos, :].reshape(p_shape) 
            else: 
                for c in range(3):
                    img[i:i+p_shape[0], j:j+p_shape[1], c] += patches[pos, c*s:c*s+s].reshape(p_shape) 
            cnt[i:i+p_shape[0], j:j+p_shape[1]] += 1
    """        
    s   = img.shape
    img = img[p_shape[0]//2:s[0]-p_shape[0]//2 - adl, p_shape[1]//2:s[1]-p_shape[1]//2 - adc]
    cnt = cnt[p_shape[0]//2:s[0]-p_shape[0]//2 - adl, p_shape[1]//2:s[1]-p_shape[1]//2 - adc]
    img[cnt!=0] = img[cnt!=0]/cnt[cnt!=0] if not rgb else img/np.tile(cnt, (3,1,1)).transpose(1,2,0)
    if return_cnt :
        return img, cnt 
    return img
 


def extract_patches_from_imgs(imgs, p_shape, per_im=None, gts=None, anno_map=False,
                              masks=None, list_given_ids=None):
    """Extract "per_im" patches randomly from each image in imgs
     
    Parameters
    ----------
    imgs : list,
        list of images of shape (height, width, n_channel) 
        (n_channel = 3 for RGB images)
    p_shape : tuple, 
             the patches shape (height, width)
    per_im : long
            number of patch per img 
    gts : list, Optional
        list of the ground-truth images of shape (height, width)
    anno_map : bool or tuple, Default False
        if True a classification map is assotiated with each patch 
        instead of one single class label. if a tuple is given, the classification
        map will be of shape the given tuple instead of "p_shape"
    mask : list, Optional
        list boolean mask images of the same shape as the input "img". 
        If given, only the pixel with true value in mask will be considered.
    
    Returns
    -------
    Patches : array-like, shape (n_samples, n_features)
        Patche vectors, where n_samples is the number of samples and
        n_features is the number of features.
    Labels : if gts is given, array-like,  
            shape (n_samples,) or (n_samples, n_features) if anno_map is True
        
    """    
    patches = []
    labels  = [] 
    for i in range(len(imgs)): 
        anno = gts[i] if gts is not None else None
        mask = masks[i] if masks is not None else None
        res = extract_patches(imgs[i], p_shape, per_im, anno_img=anno, 
                              anno_map=anno_map, mask=mask,
                                given_ids=list_given_ids[i]) 
        if gts is not None:
            patches.append(res[0])
            labels.append(res[1]) 
        else:
            patches.append(res)
    
    patches = np.concatenate([x for x in patches], axis=0) 
    if gts is not None:
        if  anno_map == True:
            labels = np.concatenate([x for x in labels], axis=0) 
        else:
            labels = np.array(labels).flatten() 
        return patches, labels
    return patches


def compute_sp_coord(sps):
    pos = []
    for l in np.unique(sps):
        pix = np.transpose(np.nonzero(sps == l))
        pos.append([pix[:,0].mean(dtype='int'), pix[:,1].mean(dtype='int')])
    return np.array(pos)

# XXX: To be updated
# - returned shape (n_samp, n_feat)
# - optimize
def slic_patches(img, anno_img=None, p_shape=(32,32), 
                       n_sp=1000, compactness=20, sigma=0, slico=True, 
                       anno_map=False, **kwargs):
    assert(img.ndim >= 2)
    
    im_shape  = img.shape
    if anno_img is not None:
        anno_img[anno_img>0] = 1 # assumes a binary segmentation framework
    im  = color.gray2rgb(img.copy()) if img.ndim == 2 else img.copy()
    sps = segmentation.slic(im, n_sp, compactness, sigma=sigma, slic_zero=slico)
    ids = np.unique(sps) 
    
    img     = np.reshape(img, (im_shape[0], im_shape[1], -1)).transpose(2,0,1)
    aux     = []
    for im in img:
        aux.append(util.pad(im, (p_shape[0]/2, p_shape[1]/2), mode='reflect'))
    img     = np.array(aux).transpose(1,2,0)
    
    patches = np.zeros((np.prod(p_shape)*img.shape[2], ids.size))
    if anno_img is not None:
        if not anno_map:
            labels = np.empty((ids.size))  
        else:
            labels = patches.copy()
            anno_img = util.pad(anno_img, (p_shape[0]/2, p_shape[1]/2), mode='reflect')
    
    sps_coord = compute_sp_coord(sps)
    sps_coord = sps_coord + np.array(p_shape)/2
    
    for k,pos in enumerate(sps_coord):
        i = pos[0]
        j = pos[1]
        patches[:, k] = img[i-p_shape[0]/2:i+p_shape[0]/2, 
                            j-p_shape[1]/2:j+p_shape[1]/2].flatten()
        if anno_img is not None:
            if not anno_map:
                labels[k]    = (anno_img[sps==k]).mean()
            else:
                labels[:, k] = anno_img[i-p_shape[0]/2:i+p_shape[0]/2, 
                            j-p_shape[1]/2:j+p_shape[1]/2].flatten()
    
    if anno_img is not None:
        return [patches, labels, sps]
    return [patches, sps]

def image_slic_patches_merging(patches, sps, p_shape):
    assert(sps.ndim == 2)
    
    img = np.zeros((sps.shape[0]+p_shape[0], sps.shape[1]+p_shape[1]))    
    cnt = np.zeros((sps.shape[0]+p_shape[0], sps.shape[1]+p_shape[1]))
    
    sps_coord = compute_sp_coord(sps)
    sps_coord = sps_coord + np.array(p_shape)/2
    
    for k,pos in enumerate(sps_coord):
        x1 = pos[0]-p_shape[0]/2
        x2 = pos[0]+p_shape[0]/2
        y1 = pos[1]-p_shape[1]/2
        y2 = pos[1]+p_shape[1]/2 
        img[x1:x2, y1:y2] += patches[:, k].reshape(p_shape)
        cnt[x1:x2, y1:y2] += 1
    
    img = img[p_shape[0]/2:img.shape[0]-p_shape[0]/2, p_shape[1]/2:img.shape[1]-p_shape[1]/2]
    cnt = cnt[p_shape[0]/2:cnt.shape[0]-p_shape[0]/2, p_shape[1]/2:cnt.shape[1]-p_shape[1]/2]
    return img, cnt

# XXX: To be updated
def slic_patches_from_imgs(imgs, gts, p_shape=(32, 32), n_sp=2000, thres=0.7,
                 compactness=0.4, anno_map=False): 
                     
    patches_ = []
    labels_  = []     
    
    for i in range(0, len(imgs)):
    
        aux = slic_patches(imgs[i], gts[i], p_shape, n_sp, 
                                 compactness=compactness, anno_map=anno_map)
        
        # try to have a balanced number of examples in each class
        # probably by thresholding the superpixels' label
        p    = aux[0]
        l    = aux[1]
        
        if not anno_map:
            cl1  = np.argwhere(l>=thres).flatten()
            cl0  = np.argwhere(l<thres).flatten()
            if len(cl1) != 0 and len(cl0) != 0: 
                rids = np.random.randint(0, len(cl0), len(cl1))
                p    = np.hstack((p[:,cl0[rids]], p[:, cl1]))
                l    = np.hstack((l[cl0[rids]], l[cl1]))
        
            l[l>thres]  = 1   
            l[l<=thres] = 0  
                                            
        patches_.append(p)
        labels_.append(l) 
    
    patches = patches_[0]
    labels  = labels_[0]
    for p in range(1, len(patches_)):
        patches = np.hstack((patches, patches_[p]))
        labels  = np.hstack((labels, labels_[p]))
        
    return [patches, labels]

 
