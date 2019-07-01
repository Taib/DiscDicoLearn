# -*- coding: utf-8 -*-
"""Usefull methods for data generation and preprocessing"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 
import os
import numpy as np
np.random.seed(100)
import pandas as pd
import functools
from skimage import io, exposure, morphology, color, transform, filters

__datasets_directory = os.path.expanduser('~/PhD/Datasets/')


def preprocess_retinal_img(img,  channel, clahe, gamma=1.7, mas=None, dilate_mask=25, luv=False):
    
    img = img[:,:, channel] if channel is not None else img          

    if img.ndim == 3 and luv:
        img = exposure.rescale_intensity(color.rgb2luv(img), out_range=(0,255)).astype('uint8') 
        
    if clahe and ( img.ndim == 2):
        img = exposure.rescale_intensity(exposure.equalize_adapthist(img), out_range=(0,255)) 
        
    elif clahe: 
        im = []
        for c in range(3):
            a = exposure.equalize_adapthist(img[:,:,c])
            im.append(exposure.rescale_intensity(a, out_range=(0,255)))
        img = np.array(im).transpose(1,2,0)
       
    if gamma:
        img = exposure.adjust_gamma(img, gamma)
        img = exposure.rescale_intensity(img, out_range=(0,1))
    
    if mas is not None:
        mas[mas != 0] = 1
        mask = morphology.dilation(mas, morphology.selem.disk(dilate_mask))
        img = img*mask if img.ndim == 2 else img*np.tile(mask, (3,1,1)).transpose(1,2,0) 

    return img

def get_data_paths(dataset_name, data_dir, f_name="test_names"):
    df = pd.read_csv("datasets/{}/{}.txt".format(dataset_name, f_name))
    x_paths = df['im_name'].map(lambda s: os.path.join(data_dir,dataset_name,s))
    y_paths = df['gt_name'].map(lambda s: os.path.join(data_dir,dataset_name,s))
    return x_paths, y_paths

def read_drive_testset(args, data_dir=''): 
    x_paths, y_paths = get_data_paths("DRIVE", data_dir)
    imgs = []
    anns = []
    mass = []
    for  x_path, y_path in zip(x_paths, y_paths):
        img = io.imread(x_path , as_grey=args['gray'] )
        ann = io.imread(y_path , as_grey=1)
        mas = io.imread('%s%s/test/mask/%s_test_mask.gif' %(data_dir, "DRIVE", x_path.split('/')[-1].split('_')[0]) , as_grey=1)
        imgs.append(img)
        anns.append(ann)
        mass.append(mas)
    return imgs, anns, mass


def read_stare_testset(args, data_dir=''): 
    x_paths, y_paths = get_data_paths("STARE", data_dir)
    imgs = []
    anns = []
    mass = [] 
    for  x_path, y_path in zip(x_paths, y_paths):
        img = io.imread(x_path , as_grey=args['gray'])
        ann = io.imread(y_path , as_grey=1)
        mas = io.imread('%s%s/mask/%s.png' %(data_dir, "STARE", x_path.split('/')[-1].split('.')[0]) , as_grey=1)
        imgs.append(img)
        anns.append(ann)
        mass.append(mas)
    return imgs, anns, mass 


def _process_pathnames(fname, lname, resize=None):  
    img = io.imread(fname)
    gt = io.imread(lname)
    if gt.ndim < 3:
        gt  = np.expand_dims(gt, -1)
    gt = gt[...,:1]
    gt = (gt > 0).astype(int) # binarize the ground-truth
    if resize is not None:
        img = transform.resize(img, resize)
        gt = transform.resize(gt, resize)
        gt = gt >= filters.threshold_otsu(gt)
    return img, gt 

### Data augmentation routines
def shift_img(img, gt, width_shift_range, height_shift_range, rotate_range): 
    if width_shift_range  or height_shift_range:
        if width_shift_range:
            width_shift_range = np.random.uniform(-width_shift_range * img.shape[1],
                                                   width_shift_range * img.shape[1])
        if height_shift_range:
            height_shift_range = np.random.uniform(-height_shift_range * img.shape[0],
                                                   height_shift_range * img.shape[0]) 
        tr = transform.AffineTransform(translation=(width_shift_range, height_shift_range )) 
        img = transform.warp(img, tr, preserve_range=True)
        if gt is not None:
            gt   = transform.warp(gt, tr, preserve_range=True) 

    if rotate_range :
        if isinstance(rotate_range, np.ScalarType):
            degre = np.random.uniform(-rotate_range,rotate_range)
        else:
            degre = np.random.uniform(rotate_range[0], rotate_range[1])
        img = transform.rotate(img, degre, preserve_range=True)
        if gt is not None:
            gt  = transform.rotate(gt, degre, preserve_range=True)
        
    return img, gt

def flip_img(img, gt, horizontal_flip, vertical_flip):
    if horizontal_flip:
        if np.random.uniform(0.0, 1.0) < 0.5 :
            img = np.flip(img, 1)
            gt =  np.flip(gt, 1) if gt is not None else gt
    if vertical_flip:
        if np.random.uniform(0.0, 1.0) < 0.5 :
            img = np.flip(img, 0)
            gt =  np.flip(gt, 0) if gt is not None else gt
    return img, gt

def _process_imgt(img, gt, gamma=0,
                clahe=False, gray=False, xyz=False,
                horizontal_flip=False, width_shift_range=0,
                height_shift_range=0, vertical_flip=0, rotate_range=0, bw_gt=True):
    img = exposure.rescale_intensity(img.astype(float), out_range=(0,1))
    if gray:
        img = color.rgb2gray(img)
    if xyz:
        img = color.rgb2xyz(img)
        img = exposure.rescale_intensity(img, out_range=(0,1))
    if clahe:
        img = exposure.equalize_adapthist(img)
    if gamma:
        img = exposure.adjust_gamma(img, gamma)
        img = exposure.rescale_intensity(img, out_range=(0,1))
    if img.ndim == 2:
        img = np.expand_dims(img, -1) 

    img, gt = flip_img(img, gt, horizontal_flip, vertical_flip)    
    img, gt = shift_img(img, gt, width_shift_range, height_shift_range, rotate_range) 
    
    return img, gt




def fixed_patch_ids_creation(im_paths, gt_paths, spatial_shape=None,
                            p_stride=16, shuffle=True, per_label=0, mask=None):    
    all_ids  = np.zeros(len(spatial_shape)+2)
    mask = mask if mask is not None else 1
    for im_path, gt_path in zip(im_paths, gt_paths) : 
        if p_stride > 0:
            ids = np.zeros(spatial_shape, dtype='int')
            if ids.ndim == 2:
                ids[0::p_stride, 0::p_stride] = 1   
            else: 
                ids[0::p_stride, 0::p_stride, 0::p_stride] = 1   
            ids = ids * mask
            ids = np.array(np.nonzero(ids)).T
            n  = len(ids)
            ap = np.c_[np.expand_dims([im_path]*n, -1), np.expand_dims([gt_path]*n, -1), ids] 
            all_ids = np.vstack((all_ids, ap)) 
        
        if per_label >0: 
            # Adding samples based on the classes distribution.
            _, gt = _process_pathnames(im_path, gt_path, resize=spatial_shape) 
            cls_ids = []
            for c in np.unique(gt):
                search_area = np.nonzero((np.squeeze(gt) == c) * mask)
                if len(search_area[0]) == 0:
                    continue
                search_area = np.array(search_area).T
                search_area = np.random.permutation(search_area) 
                cls_ids.append(search_area[:per_label])
            cls_ids = np.concatenate([x for x in cls_ids])  
            n  = len(cls_ids)
            ap = np.c_[np.expand_dims([im_path]*n, -1), np.expand_dims([gt_path]*n, -1), cls_ids] 
            all_ids = np.vstack((all_ids, ap)) 

    all_ids = all_ids[1:]
    if shuffle:
        np.random.shuffle(all_ids)  
    return all_ids 


class Patch_Sequence:
    def __init__(self, fixed_patch_ids, p_shape=(32,32,3),
                reader_fn=functools.partial(_process_pathnames),
                preproc_fn=functools.partial(_process_imgt),
                norm_fn=None,
                batch_size=32,
                anno_map=False,
                MAX_IM_QUEUE=20, unsup=False):
        self.ids = fixed_patch_ids #
        self.p_shape = p_shape
        self.batch_size = batch_size
        self.reader_fn = reader_fn
        self.preproc_fn = preproc_fn
        self.MAX_IM_QUEUE = MAX_IM_QUEUE
        self.im_stack = {}
        self.norm_fn = norm_fn
        self.anno_map = anno_map

    def __len__(self):
        if self.ids is not None:
            return int(np.ceil(len(self.ids) / float(self.batch_size)))
        return -1
    
    def __getitem__(self, idx):
        cur_id = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size] 

        batch_x = []
        batch_y = []
        for pos in cur_id: 
            pid, pim, pgt = pos[2:], pos[0], pos[1] 
            x_p, y_p = pid.astype(int) 
            hash_im = hash(pim)
            if not self.im_stack.has_key(hash_im):
                img, gt = self.reader_fn(pim, pgt) 
                img, gt = self.preproc_fn(img, gt) 
                img = np.pad(img, ((self.p_shape[0]//2,), (self.p_shape[1]//2,), (0,)), mode='reflect')
                gt = np.pad(gt, ((self.p_shape[0]//2,), (self.p_shape[1]//2,), (0,)), mode='reflect')  
                if len(self.im_stack.keys()) > self.MAX_IM_QUEUE:
                    self.im_stack.popitem()
                self.im_stack[hash_im] = (img, gt)
            else:
                img, gt = self.im_stack[hash_im] 
            patch = img[x_p:x_p+self.p_shape[0], y_p:y_p+self.p_shape[1]].flatten()
            if not self.anno_map:
                label = gt[x_p+self.p_shape[0]//2, y_p+self.p_shape[1]//2] 
            else:
                label = gt[x_p:x_p+self.p_shape[0], y_p:y_p+self.p_shape[1]].flatten()
            batch_x.append(patch)
            batch_y.append(label)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_x = self.norm_fn(batch_x) if self.norm_fn is not None else batch_x
        return batch_x, batch_y


def patch_dataset(dataset_name, f_name, image_shape, p_shape, batch_size=1, gamma=0., 
            clahe=False,  gray=False, xyz=False,
            p_stride=32, per_label=0, shuffle=True,
            width_shift_range=0, height_shift_range=0, 
            horizontal_flip=False,vertical_flip=False,
            rotate_range=0, MIN_PATCH_STD=None, MAX_IM_QUEUE=100, norm_fn=None,
            anno_map=False,
            data_dir=__datasets_directory):
 
    x_train_paths, y_train_paths = get_data_paths(dataset_name, data_dir, f_name=f_name)

    dataset_ids = fixed_patch_ids_creation(x_train_paths, y_train_paths, spatial_shape=image_shape[:2],
                                        p_stride=p_stride, per_label=per_label,
                                        shuffle=shuffle) 



    prepro_cfg = dict(gamma=gamma, horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip, width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range, clahe=clahe, gray=gray, xyz=xyz)
    prepro_fn = functools.partial(_process_imgt, **prepro_cfg) 

    reader_cfg = dict(resize=image_shape[:2])
    reader_fn = functools.partial(_process_pathnames, **reader_cfg)

    generator =  Patch_Sequence(dataset_ids, p_shape=p_shape,
                                reader_fn=reader_fn, preproc_fn=prepro_fn, norm_fn=norm_fn,
                                batch_size=batch_size, anno_map=anno_map, MAX_IM_QUEUE=MAX_IM_QUEUE)

    x, y = [], []
    for i in range(len(generator)):
        b_x, b_y = generator[i]
        x.append(b_x)
        y.append(b_y)
    return np.concatenate(x), np.squeeze(np.concatenate(y))