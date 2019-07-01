# -*- coding: utf-8 -*-
"""Quick test for retinal blood vessels' segmentation"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 

import os
import time
import numpy as np 
import models.meths_dl_models as mtd
import pickle
import functools

from skimage import  morphology, filters, io, exposure

from ret_datagen import patch_dataset, _process_pathnames, _process_imgt, get_data_paths, fixed_patch_ids_creation, Patch_Sequence
#import matplotlib.pyplot as plt

import models.funcs as utils_funcs
import utils.miscs as utils_miscs
from utils.tvl1 import solve_TVL1
from utils.extracts import  image_patches_merging

from joblib import Parallel, delayed

def testset_test(meth, args,data_dir='', tvpar=[0.8], save_path='',
                 m4bin=5, stride=1):
    accs  = {}
    for t in tvpar:
        accs[str(t)] = [] 
    times = []
    
    if save_path != "" and not os.path.exists(save_path):
        os.makedirs(save_path)   
    
    x_test_paths, y_test_paths = get_data_paths(args['dataset_name'], data_dir, f_name='test_names')
    for x_path, y_path in zip(x_test_paths, y_test_paths):
        im, gt = _process_pathnames(x_path, y_path, args['image_shape'][:2])
        im, gt = _process_imgt(im, gt, gamma=args['gamma'], clahe=args['clahe'], gray=args['gray'])
        if args['dataset_name'] == "DRIVE":
            mask_path = '%s%s/test/mask/%s_mask.gif' %(data_dir,  args['dataset_name'], x_path.split('/')[-1].split('.')[0]) 
        else:
            mask_path = '%s%s/mask/%s.png' %(data_dir,  args['dataset_name'], x_path.split('/')[-1].split('.')[0]) 
        _, mask= _process_pathnames(y_path, mask_path,  args['image_shape'][:2])

        mask = np.squeeze(mask)
        mask = morphology.binary_erosion(mask, selem=morphology.selem.disk(8))

        pix_ids = fixed_patch_ids_creation([x_path], [y_path], spatial_shape=args['image_shape'][:2], p_stride=1, shuffle=False, per_label=0, mask=mask)

        prepro_cfg = dict(gamma=args['gamma'], clahe=args['clahe'], gray=args['gray'])
        prepro_fn = functools.partial(_process_imgt, **prepro_cfg) 

        reader_cfg = dict(resize=args['image_shape'][:2])
        reader_fn = functools.partial(_process_pathnames, **reader_cfg)     
    
        generator =  Patch_Sequence(pix_ids, p_shape=args['p_shape'],
                                    reader_fn=reader_fn, preproc_fn=prepro_fn, norm_fn=utils_funcs.norm_cols,
                                    batch_size=args['batch_size'], MAX_IM_QUEUE=100)

        pre_fov = Parallel(n_jobs=args['n_jobs'])(delayed(meth.predict)(generator[i][0]) for i in range(len(generator)))
        pre_fov = np.concatenate([p.tolist() for p in pre_fov])
        if isinstance(meth, mtd.InputToMap):  
            ids     = np.zeros(mask.shape)
            ids[0:-1:1, 0:-1:1] = 1
            ids    *= mask 
            ids     = np.squeeze(np.nonzero(ids.flatten())) 
            prob    = image_patches_merging(pre_fov, mask.shape, args['p_shape'], given_ids=ids) 
            prob    = prob*mask  
        else:
            prob = np.zeros(mask.shape).flatten() 
            prob[np.squeeze(np.nonzero(mask.flatten()))] = pre_fov
            prob = prob.reshape(mask.shape)
        if save_path != "":     
            out_name = save_path + '/' 
            out_name+= x_path.split('/')[-1].split('.')[0]
            io.imsave(out_name + '.png', exposure.rescale_intensity(prob, out_range=(0,255)).astype(int))
            for t in tvpar:
                if t != "notv":
                    pre, E = solve_TVL1(prob, t, 100)
                    pre = np.where(pre > filters.threshold_otsu(pre, m4bin), 1, 0) 
                else:
                    pre = np.where(prob > filters.threshold_otsu(prob, m4bin), 1, 0) 
                acc = utils_miscs.seg_metrics(pre, np.squeeze(gt))
                accs[str(t)].append(acc)  
                io.imsave(out_name + '_tv_%s.png'%(t), pre.astype(int)*255 )  
      
    for t in tvpar:
        accs[str(t)] =  [np.array(accs[str(t)]), np.array(accs[str(t)]).mean(0)]
        print (save_path, 'Average metrics : ', 'tv: ', t, ' ===> ',  accs[str(t)][1])
    return accs


def load_data(args): 
    # Data preparation 
    p_shape = args['p_shape'] + (1,) if  args['gray'] else  args['p_shape']+(3,)
    f_name = "train_names"

    X_train, y_train = patch_dataset(args['dataset_name'], f_name, args['image_shape'], p_shape, batch_size=args['batch_size'], gamma=args['gamma'], 
                                    clahe=args['clahe'],  gray=args['gray'], per_label=3000, shuffle=True, norm_fn=utils_funcs.norm_cols, anno_map=args['anno_map'])

    
    return [X_train, y_train]
 
if __name__ == '__main__' : 
    datasets_directory = os.path.expanduser('~/PhD/Datasets/')
    
    method_id    = ["FVP"]
    

    args = {}
    args['normal_m']     = 0  
    args['gray']         = 1
    args['channel']      = None
    args['clahe']        = True
    args['gamma']        = 0
    args['p_shape']      = (16,16)
    args['n_atoms']      = 500  
    args['n_iter']       = 1000
    args['sc_mode']      = 2
    args['lambda2']      = 0.0   
    args['rand_init']    = False
    args['init_iter']    = 1000
    args['n_jobs']       = 8
    args['image_shape']  = (584, 564, 1)
    args['batch_size']   = 60000
    args['anno_map']     = False


    
    for dname in ["DRIVE"]:
        args['image_shape']  = (604, 700, 1) if dname == 'STARE' else args['image_shape']
        args['dataset_name'] = dname 
        data = load_data(args)
        args['anno_map']     = False

        
        for n in method_id:
            tvparam = [.5, .8, .9, 1., 1.3, 1.5, 1.8]

            if n == 'DLCL': 
                params = args.copy()   
                params['lambda1'] = 0.5
                params['n_atoms'] = 1000
                meth = mtd.DLCL() 
                mtd.set_attributes(meth, params) 

            elif n == "DPC":   
                params = args.copy()  
                params['eta'] = 0.26
                params['kappa'] = 0.0
                params['lambda1'] = 0.5
                params['dico_algo'] = 'am'
                params['n_iter']    = 0
                meth = mtd.DPC() 
                mtd.set_attributes(meth, params) 
            elif n == "JDCL": 
                params = args.copy()  
                params['beta2'] =  0.26
                params['beta1'] = 0.0
                params['lambda1'] = 0.58
                meth = mtd.JDCL()   
                mtd.set_attributes(meth, params) 
            elif n == "FVP":
                args['anno_map']     = True  
                data = load_data(args)
                params = args.copy() 
                params['n_atoms'] = [1000, 1000]
                params['lambda2'] = 0.0
                params['beta1']  = 0.03
                params['beta2']  = 0.03
                params['lambda1'] = 0.03
                params['technic'] = 5
                params['proj_tech'] = 0 
                meth = mtd.SRMC() 
                mtd.set_attributes(meth, params) 
                data[1] = utils_funcs.norm_cols(data[1].astype(float)) 
                tvparam = ['notv']

            pref = 'ddl_{}_meth{}_atoms{}_iter{}_sc{}_lambda{}'.format(dname, n, params['n_atoms'], params['n_iter'], params['sc_mode'], params['lambda1'])

            if os.path.exists(pref+".npy"):
                print('[Passing] Model path exists: ', pref+".npy")
                meth = np.load(pref+".npy", allow_pickle=True).item()   
            else:
                meth.fit(data[0], data[1])
                np.save(pref+".npy", [meth])  
            accs = testset_test(meth, params, datasets_directory, tvpar=tvparam, save_path=pref, m4bin=10)
            np.save(pref+"_accs.npy", [accs])
               