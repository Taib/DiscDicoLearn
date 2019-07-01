# -*- coding: utf-8 -*-
"""Cross-validating methods on retinal images"""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr>
  
import os
import numpy as np 
import pandas as pd
import models.ddl_models as mtd 
from sklearn.model_selection import RandomizedSearchCV
#import matplotlib.pyplot as plt

import models.funcs as utils_funcs
from ret_datagen import patch_dataset



def load_data(args): 
    # Data preparation 
    p_shape = args['p_shape'] + (1,) if  args['gray'] else  args['p_shape']+(3,)
    f_name = "train_names"

    X_train, y_train = patch_dataset(args['dataset_name'], f_name, args['image_shape'], p_shape, batch_size=args['batch_size'], gamma=args['gamma'], 
                                    clahe=args['clahe'],  gray=args['gray'], per_label=500, shuffle=True, norm_fn=utils_funcs.norm_cols)

    
    return [X_train, y_train]
 
if __name__ == '__main__' : 
    datasets_directory = os.path.expanduser('~/PhD/Datasets/')
    
    method_id    = ['JDCL', "DLCL", 'DPC', 'FVP']
    

    args = {}
    args['normal_m']     = 0  
    args['gray']         = True
    args['channel']      = None
    args['clahe']        = True
    args['gamma']        = 1.7
    args['p_shape']      = (9,9)
    args['n_atoms']      = 1000  
    args['n_iter']       = 1000
    args['sc_mode']      = 2
    args['lambda2']      = 0.0   
    args['rand_init']    = True
    args['n_jobs']       = 30
    args['image_shape']  = (584, 568, 1)
    args['batch_size']   = 60000 
    args['dataset_name'] = "DRIVE"  
    
    data = load_data(args)
    
    for n in method_id:
        pref = 'crossvalidation_dr_meth{}_atoms{}_iter{}_sc{}.txt'.format(n, args['n_atoms'], args['n_iter'], args['sc_mode'])
        params = {}
        params['lambda1'] = np.arange(0,1, 0.01 ) 

        if n == 'DLCL': 
            meth = mtd.DLCL() 
            mtd.set_attributes(meth, args) 
        elif n == "DPC":    
            params['eta'] = np.arange(0,1, 0.01)
            params['kappa'] = np.arange(0,1, 0.01) 
            meth = mtd.DPC(dico_algo="am", n_iter=1000) 
            mtd.set_attributes(meth, args) 
        elif n == "JDCL":
            params['beta1'] = np.arange(0,1, 0.01)
            params['beta2'] = np.arange(0,1, 0.01 )
            meth = mtd.JDCL()   
            mtd.set_attributes(meth, args) 
        elif n == "FVP":
            args['anno_map']     = True  
            data = load_data(args)
            params['beta1'] = np.arange(0,1, 0.01)
            params['beta2'] = np.arange(0,1, 0.01)
            args['technic'] = 5  
            args['n_atoms'] = [1000, 1000]
            meth = mtd.SRMC() 
            mtd.set_attributes(meth, args)
            data[1] = utils_funcs.norm_cols(data[1].astype(float)) 
        
        rscv = RandomizedSearchCV(meth, params, cv=2, n_jobs=30, n_iter=100)
        rscv.fit(data[0], data[1])
        
        results = rscv.cv_results_
        df = pd.DataFrame(results, columns=results.keys())
        df.to_csv(pref)