# -*- coding: utf-8 -*-
"""Classes of all Discriminative Dictionary learning based models."""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 

import sys
import numpy as np
import time
import h5py
from scipy import linalg, sparse
from abc import ABCMeta 
import utils_dl_optim
import utils_funcs 

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors as KNN


def set_attributes(meth, params):
    assert(isinstance(params, dict))
    assert((isinstance(meth, InputToLabel) or isinstance(meth, InputToMap))) 
    for k in params.keys():
        if k in meth.__dict__.keys():
            meth.__setattr__(k, params[k])


def init_sub_dictionaries(X, rand_init=False, n_atoms=100, lambda1=0.1,
                         sc_mode=2, init_iter=100): 
    D = []
    for i in range(len(X)): 
        if rand_init:
            ids = np.random.permutation(range(X[i].shape[0]))[:n_atoms]
            Di  = X[i][ids]
            Di  = utils_funcs.norm_cols(Di)
        else: 
            Di, _ =utils_dl_optim.odl(X[i].T, n_atoms, lambda1,
                                 mode=sc_mode, n_iter=init_iter) 
            Di = Di.T
        D.append(Di) 
    return D
 
    
class InputToLabel( BaseEstimator, ClassifierMixin):  
    """Mixin class for all Dictionary learning classifiers.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    __metaclass__ = ABCMeta
 
    def score(self, X, y):
        pred = self.predict(X)  
        return (pred == y).mean()
     
    
    def save_model(self, path):
        path = path + '.h5' if not path.endswith('.h5') else path
        with h5py.File(path, 'w') as hf:
            for attr in self.__dict__.keys():
                d = self.__getattribute__(attr)  
                if d is None:
                    continue
                dtype = type(d)
                hf.create_dataset(attr,  data=self.__getattribute__(attr), 
                                  dtype=dtype)
    
    def load_model(self, path):
        path = path + '.h5' if not path.endswith('.h5') else path
        with h5py.File(path, 'r') as hf:
            for attr in self.__dict__.keys():  
                if attr in hf.keys():
                    self.__setattr__(attr, hf[attr].value) 
    
    def model_conv(self, X, old_D=None):
        e = self.rec_error(X)
        c = None
        if old_D is not None:
            c = np.sqrt(((self.D-old_D)**2).sum())/np.sqrt((old_D**2).sum())
        return [e, c]

class InputToMap(BaseEstimator, RegressorMixin): 
    """Mixin class for all Dictionary learning regressors.
    
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    __metaclass__ = ABCMeta
  
    def save_model(self, path):
        path = path + '.h5' if not path.endswith('.h5') else path
        with h5py.File(path, 'w') as hf:
            for attr in self.__dict__.keys():
                d = self.__getattribute__(attr)  
                if d is None:
                    continue
                hf.create_dataset(attr,  data=self.__getattribute__(attr))
    
    def load_model(self, path):
        path = path + '.h5' if not path.endswith('.h5') else path
        with h5py.File(path, 'r') as hf:
            for attr in self.__dict__.keys():  
                if attr in hf.keys():
                    self.__setattr__(attr, hf[attr].value) 
    
    def score(self, X, Y):
        pred = self.predict(X)
        s = (np.abs(pred - Y)).mean() 
        return s 
     


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




class DLCL(InputToLabel): 
    """
     Sparse coding and classifier
        
     
     This class first learns a dictionary, then used the sparse code produced
     to train a classifier
     
    Parameters
    ----------
    n_atoms : integer, Default 100
              the number of atoms () 
    n_iter : integer, Default 100
             the number of iterations 
    lambda1 : float, Default 0.1
            the weight or the constraint on the sparsity term 
    sc_mode : integer in {0, 1, 2, 3, 4, 5}, Default 2
            the type of lasso formulation to use.
            - 0 : the lasso formulation,
            - 1 : minimize the sparsity s.t. small noise,
            - 2 : the basis pursuit denoising formulation,
            - 3 : l_0 version of the lasso,
            - 4 : l_0 version of the case 1,
            - 5 : l_0 version of the basis pursuit denoising.
    pos_coef : bool, Default False
            positivy constrain on the sparse codes 
    pos_dico : bool, Default False
            positivy constrain on the atoms' values 
    rand_init : bool, Default 
        whether or not use a random initialization of the dictionary
    init_iter : int, Default
        number of iterations to perform in initialization per dictionary
     
    Returns
    -------
    self : object
        Returns self. 
    """
    def __init__(self, n_atoms=80, lambda1=0.1, sc_mode=2, n_iter=20, lambda2=0, 
                 dico_algo='on', clf_algo='rf', rand_init=False, init_iter=100,
                 init_D=None, weighted_lasso=False, min_weight=10e-3, 
                 return_model=False, n_jobs=8, pos_coef=False, pos_dico=False,
                 kernel_f='poly', init_L=None, init_Lc=None, init_K=None): 
                 
        self.name      = 'DLCL' 
        
        self.dico_algo = dico_algo
        self.sc_mode   = sc_mode
        
        self.n_atoms   = n_atoms
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.n_iter    = n_iter
        self.n_jobs    = n_jobs
        
        self.pos_coef  = pos_coef
        self.pos_dico  = pos_dico
        
        self.model     = None  
        self.return_model= True
        self.l_time    = 0
         
        self.rand_init = rand_init
        self.init_iter = init_iter
        
        self.clf_algo  = clf_algo
        self.clf       = None
        
        self.L         = init_L
        self.Lc        = init_Lc
        
        self.K         = init_K
        self.kernel_f  = kernel_f
         
        self.weighted_lasso= weighted_lasso
        self.min_weight    = min_weight
        
        self.short_print = '%s__%d__%d__%d__%d__%d__%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.weighted_lasso,
                         self.pos_coef, self.sc_mode, self.lambda1)
        self.full_print= '%s__atoms_%d__iter_%d__wl_%d__pcoef_%d__scmode_%d__lambda_%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.weighted_lasso,
                          self.pos_coef, self.sc_mode, self.lambda1)
                          
    def __str__(self):
        s=  'DLCL --- Optimization \n'
        s+= 'n_atoms:%d __ n_iter:%d  \n' %(self.n_atoms, self.n_iter)
        s+= 'lambda1:%.2f __ lambda2:%.2f   \n' %(self.lambda1, self.lambda2) 
        s+= 'dico_algo:%s __ sc_mode:%s \n' %(self.dico_algo, self.sc_mode)
        return s
    
    
    def fit(self, X, y):
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X. 
            
        Returns
        -------
        self : object
            Returns self.
        """ 
     
        print(self)
        
        params = {'n_atoms':self.n_atoms, 'lambda1':self.lambda1, 
                  'n_iter':self.n_iter, 'model':self.model, 
                  'return_model':self.return_model }

        tic = time.time() 
        if self.dico_algo == 'on': 
            params['mode']     = self.sc_mode
            params['lambda2']  = self.lambda2
            params['pos_coef'] = self.pos_coef
            params['pos_dico'] = self.pos_dico   
            params['n_jobs']   = self.n_jobs             
            self.D, self.model = utils_dl_optim.odl(X.T, **params )
            self.D = self.D.T
            
        elif self.dico_algo == 'ks': 
            self.D, self.modell =utils_dl_optim.ksvd(X, **params )
                                        
        elif self.dico_algo == 'gr':  
            if self.Lc is None:
                self.Lc =utils_dl_optim.manifold_laplacian(y)  
            
            params['Lc']       = self.Lc
            params['L']        = self.L
            self.D, self.model = utils_dl_optim.graphDL(X, **params)
        
        
        elif self.dico_algo == 'ke': 
            
            self.trainX = np.copy(X)
            if self.K is None:
                self.K = utils_dl_optim.kernel_matrix(X)
            
            
            params['K']        = self.K
            self.D, self.model = utils_dl_optim.kernelDL(self.K, **params)
        
        
        A = self.decode(X) 
        
        self.clf = utils_funcs.classifier(A, y, name=self.clf_algo, n_jobs=self.n_jobs)
    
        self.l_time = (time.time() -tic)
    
    def decode(self, X):        
        """
        Find the sparse representation of each sample of X over 
        the dictionary of this class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        A : array-like, shape (n_samples, n_atoms)
            Sparse representations of the samples in X.
        """ 
        W = None
        if self.weighted_lasso:
            W = []
            for p in range(self.n_atoms):
                w = X - np.tile(self.D[p,:], (X.shape[0],1)).T
                w = np.sqrt((w**2).sum(0))
                W.append(w)
            W = np.array(W, order='FORTRAN')
            W = np.where(W==0, self.min_weight, W)
        
        params = { 'mode':self.sc_mode, 'lambda1':self.lambda1, 'W':W,
                  'lambda2':self.lambda2, 'n_jobs':self.n_jobs, 
                  'pos_coef':self.pos_coef, 'K':self.K, 
                  'Lc':self.Lc, 'L':self.L }
        
        if self.dico_algo != 'ke':          
            A =utils_dl_optim.sparse_decode(X.T, self.D.T, **params).T
        else:
            A = []
            append = A.append
            for i in range(X.shape[0]):
                p =utils_dl_optim.kernel_project(X[i,:], self.trainX )
                append(utils_dl_optim.komp(p, self.D, self.K, self.lambda1))
                
        return A
     
    def rec_error(self, X):
        """
        Compute the reconstruction error of X over 
        the dictionary of this class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        e : float,
             result of \|X - AD\|_F^2.
        """ 
        Xh = self.reconstruct(X)
        return 0.5*((X - Xh)**2).sum()
    
    def reconstruct(self, X):        
        """
        Reconstruct each sample of X using the dictionary of this class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        r : array-like, shape (n_samples, n_features),
          The reconstruct each sample of X using the dictionary.
        """ 
        A = self.decode(X)
        return np.dot(A, self.D)
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        Examples
        -------   
        >>> import numpy as np 
        >>> m = 16;n = 20;n_classes = 2;n_atoms=32;
        >>> np.random.seed(0)
        >>> X = [np.random.randn(m,n) for i in range(n_classes)]
        >>> Y = [np.ones(n)*i for i in range(n_classes)]
        >>> meth = DLCL(n_atoms)
        >>> meth.fit(np.c_[X[0], X[1]])
        >>> meth.predict(X[0])        
        """ 
        A = self.decode(X) 
        return self.clf.predict(A)
    
    def predict_proba(self, X):
        """
        Return the probability estimates on samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        A = self.decode(X)
        return self.clf.predict_proba(A)
     

    
class MultiMethod1(InputToLabel):
    def __init__(self, n_atoms=80, n_level=2, lambda1=0.1, sc_mode=2, n_iter=20, 
                 lambda2=0, dico_algo='on', clf_algo='rf', rand_init=False,
                 init_iter=100, init_D=None, weighted_lasso=False, min_weight=10e-3, 
                 return_model=False, n_jobs=8, pos_coef=False, pos_dico=False,
                 kernel_f='poly', init_L=None, init_Lc=None, init_K=None,):
        self.n_level = n_level
        self.meths = []
        for i in range(self.n_level):
            m = DLCL(self, n_atoms, lambda1, sc_mode, n_iter, lambda2, 
                 dico_algo, clf_algo, rand_init, init_iter,init_D, 
                 weighted_lasso, min_weight, return_model, n_jobs, pos_coef, 
                 pos_dico, kernel_f, init_L, init_Lc, init_K)
            self.meths.append(m)
    
    def fit(self, X, y):
        rec = X.copy()
        for i in range(self.n_level):
            self.meths[i].fit(rec, y)
            rec = self.meths[i].reconstruct(rec)
    
    def predict(self, X):
        pred = [] 
        rec  = X.copy()
        for i in range(self.n_level):
            pred.append(self.meths[i].predict_proba(rec))
            rec = self.meths[i].reconstruct(rec)
        pred = np.argmax(np.array(pred).max(0), 1)
        return pred



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

 
 
class DPC(InputToLabel): 
    """
    Patch classification based on incoherent dictionaries as proposed in the paper\n
    @paper: Ramirez, I., Sprechmann, P., and Sapiro, G. (2010). 
            Classification and clustering via dictionary learning with structured 
            incoherence and shared features.
            In in IEEE Conference on Computer Vision and Pattern Recognition(CVPR).
     
    Parameters
    ----------
    n_atoms : integer, Default 100
              the number of atoms () 
    eta : float, Default 0.1
        the weight of the incoherence term
    kappa : float, Default 0.1
        the weight of the discrimination term 
    lr : float, Default computed from the data
        the learning rate used if 'dico_algo' equals "sgd"
    inco_thres:  float, Default 0.95
        the threshold value used to remove shared atoms, see the paper
    n_iter : integer, Default 100
             the number of iterations 
    lambda1 : float, Default 0.1
            the weight or the constraint on the sparsity term 
    sc_mode : integer in {0, 1, 2, 3, 4, 5}, Default 2
            the type of lasso formulation to use.
            - 0 : the lasso formulation,
            - 1 : minimize the sparsity s.t. small noise,
            - 2 : the basis pursuit denoising formulation,
            - 3 : l_0 version of the lasso,
            - 4 : l_0 version of the case 1,
            - 5 : l_0 version of the basis pursuit denoising.
    pos_coef : bool, Default False
            positivy constrain on the sparse codes 
    pos_dico : bool, Default False
            positivy constrain on the atoms' values 
    rand_init : bool, Default 
        whether or not use a random initialization of the dictionary
    init_iter : int, Default
        number of iterations to perform in initialization per dictionary
     
    Returns
    -------
    self : object
        Returns self. 
    """
    def __init__(self, n_atoms=80, eta=0.1, kappa=0.0, lr=0, inco_thres=0.95,
                 lambda1=0.1, sc_mode=2, n_iter=20, lambda2=0, 
                 dico_algo='sgd', rand_init=False, init_iter=100,
                 init_D=None, weighted_lasso=False, min_weight=10e-3, 
                 return_model=False, n_jobs=8, pos_coef=False, pos_dico=False): 
                 
        self.name      = 'DPC' 
        
        self.eta       = eta
        self.kappa     = kappa
        self.lr        = lr 
        self.inco_thres= inco_thres 
        
        self.dico_algo = dico_algo
        self.sc_mode   = sc_mode
        
        self.n_atoms   = n_atoms
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.n_iter    = n_iter
        self.n_jobs    = n_jobs 
        
        self.pos_coef  = pos_coef
        self.pos_dico  = pos_dico
        
        self.D         = init_D
        self.model     = None  
        self.return_model= True
        self.l_time    = 0

        self.n_cls = -1 
         
        self.rand_init = rand_init
        self.init_iter = init_iter
        
        self.weighted_lasso= weighted_lasso
        self.min_weight    = min_weight 
        
        self.short_print = '%s__%d__%d__%.3f__%.3f__wl_%d__%d__%d__%d__%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.eta, self.kappa, self.weighted_lasso,
                          self.rand_init, self.pos_coef, self.sc_mode, self.lambda1)
        self.full_print= '%s__atoms_%d__iter_%d__eta_%.3f__kappa_%.3f__wl_%d__rinit_%d__pcoef_%d__mlass_%d__lambda_%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.eta, self.kappa, self.weighted_lasso,
                          self.rand_init, self.pos_coef, self.sc_mode, self.lambda1)
    
    
    def __str__(self):
        s= 'DPC (Based on Ramirez2010) --- Optimization \n'
        s+= 'n_atoms:%d __ n_iter:%d __ rand_init:%d \n' %(self.n_atoms, self.n_iter, self.rand_init)
        s+= 'lambda1:%.3f __ sc_mode:%d \n' %(self.lambda1, self.sc_mode)
        s+= 'eta:%.3f __ kappa:%.3f __ thres:%.3f \n' %(self.eta, self.kappa, self.inco_thres)
        s+= 'dico_algo:%s __ cls:%s \n' %(self.dico_algo, self.n_cls)
        return s
        
    def fit(self, data, labels): 
        
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        labels : array-like, shape (n_samples,)
            Target vector assosiated with X. 
            
        Returns
        -------
        self : object
            Returns self.
        """ 
        self.n_cls = np.unique(labels).astype('int')
        assert(len(self.n_cls) >= 2)
         
        print(self)
        
        X = [data[labels==i, :] for i in self.n_cls]  
        
        if self.D is None : 
            self.D = init_sub_dictionaries(X, self.rand_init, self.n_atoms, 
                                            self.lambda1, self.sc_mode, 
                                            self.init_iter)  
        #A = [self.decode(X[i])[i] for i in self.n_cls] 


        tic  = time.time() 
        for ii in range(self.n_iter):
            print 'iteration: %d' %ii 
            
            rand_idx = np.arange(len(X))
            np.random.shuffle(rand_idx)
            
            for i in rand_idx: 
                # Update code
                A[i] = self.decode(X[i])[i]
                # Update dictionary
                self.dico_update(A[i], X, i)
         
        # Dealing with the sharing atoms     
        #self.threshold()
        
        self.l_time = time.time() - tic
     
    def dico_update(self, A, X, i):
        others = np.arange(len(self.D))
        others = np.delete(others,i)
        
        P = np.zeros((self.D[i].shape[1], self.D[i].shape[1]))
        R = np.zeros((self.D[i].shape[0], self.D[i].shape[0]))
        Q = 0
        S = 0
        
        if self.eta != 0: 
            for j in others:
                P = P + np.dot(self.D[j].T, self.D[j])
            Q  = np.dot(self.D[i], P)
        
        if self.kappa != 0:
            for k in others:
                Aj = self.decode(X[k])[i]
                R   = R + np.dot(Aj.T, Aj)
            S = np.dot(R, self.D[i])
        if self.dico_algo == 'sgd':
            """
            Stochastic gradient descent
            """
            # ----- learning rate from the beta-smoothness lemma ----- 
            beta = np.sqrt((np.dot(A.T, A)**2).sum()) 
            beta += np.sqrt(self.eta*(np.array(P)**2).sum())
            beta += np.sqrt(self.kappa*(np.array(R)**2).sum())
            
            lr   = 1./beta if self.lr == 0 else self.lr 
            
            self.D[i] = self.D[i] + lr * (np.dot(A.T, (X[i] - np.dot(A, self.D[i]))) - self.eta*Q - self.kappa*S)
            
        elif self.dico_algo == 'am':
            """
            Alternative minimization: D(AA^T + \kappa R) + \eta PD =  XA^T
            """ 
            self.D[i] = linalg.solve_sylvester(np.dot(A.T, A) + self.kappa*R, self.eta*P, np.dot(A.T, X[i]))
            
        self.D[i] =utils_funcs.norm_cols(self.D[i])
        
        return Q
    
    def threshold(self):
        if self.eta == 0.:
            return 0
        
        for i in range(len(self.D)):
            for j in range(i+1, len(self.D)):
                gram = np.dot(self.D[i], self.D[j].T)
                pos  = np.argwhere(gram >= self.thres)
                
                self.D[i][np.unique(pos[:,0]), :] = 0
                self.D[j][np.unique(pos[:,1]), :] = 0
    
    def compute_inco(self):
        inco = np.zeros((len(self.D), len(self.D)))
        for i in range(len(self.D)):
            for j in range(len(self.D)):
                inco[i, j] = np.sqrt((np.dot(self.D[i], self.D[j])**2).sum())
        return inco       
    
    def decode(self, X):   
        """
        Find the sparse representation of each sample of X over 
        the learned dictionaries of this class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        A : list, shape (n_cls, ) 
            where n_cls is the number of classes/dictionaries.
            Each element of this list is a matrix of shape (n_atoms, n_samples).
        """ 
        W = []
        for i in range(len(self.D)):
            Wi = None
            if self.weighted_lasso:
                Wi = []
                for p in range(self.D[i].shape[0]):
                    w = X - np.tile(self.D[i][p,:], (X.shape[0],1)).T
                    w = np.sqrt((w**2).sum(0))
                    Wi.append(w)
                Wi = np.array(Wi, order='FORTRAN')
                Wi = np.where(Wi==0, self.min_weight, Wi)
            W.append(Wi)
        
        params = { 'sc_mode':self.sc_mode, 'lambda1':self.lambda1, 
                  'lambda2':self.lambda2, 'n_jobs':self.n_jobs,  
                  'pos_coef':self.pos_coef}
        A = []
        for i in self.n_cls:
            params['W'] = W[i]
            params['D'] = self.D[i].T 
            Ai =utils_dl_optim.sparse_decode(X.T , **params).T
            A.append(Ai)                               
        return A
    
    def model_conv(self, X, old_D=None):
        e = self.rec_error(X)
        c = None
        if old_D is not None:
            Dt2 = np.array(self.D).transpose(1, 0, 2).reshape((self.D.shape[0], -1))
            c = np.sqrt(((Dt2-old_D)**2).sum())/np.sqrt((old_D**2).sum())
        return [e, c, self.compute_inco()]
    
    def rec_error(self, X):
        Xa = self.reconstruct(X)
        e = [0.5*((X - Xa[i])**2).sum() for i in self.n_cls]
        return e
        
    def reconstruct(self, X):
        A = self.decode(X)
        return [np.dot(A[i], self.D[i]) for i in range(len(A))]
    
    def predict(self, X): 
        """
        Perform classification on samples in X.
        
        The class of the dictionary that best represents each sample is
        returned.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.    
        """  
        errs = []
        A    = self.decode(X)
        for i in self.n_cls:
            er = 0.5*((X - np.dot(A[i], self.D[i]))**2).sum(1)**2 
            #er += self.lambda1*np.abs(A[i]).sum(1) if self.sc_mode == 2 else er
            errs.append(er) 
        return np.argmin(errs, 0)
        
    def predict_proba(self, X):  
        errs = []
        A    = self.decode(X)
        for i in self.n_cls:
            er = 0.5*((X - np.dot(A[i], self.D[i]))**2).sum(1)**2 
            er += self.lambda1*np.abs(A[i]).sum(1) if self.sc_mode == 2 else er
            errs.append(er)
            
        return utils_funcs.softmax(errs, 0)
        



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



class JDCL(InputToLabel):
    """
    Patch classification based on the Discriminative K-SVD (Zhang et al. 
    CVPR2010) or the Label-consistent K-SVD (Jiang et al)
    
    Parameters
    ----------
    n_atoms : integer, Default 100
              the number of atoms () 
    beta1 : float, Default 0.1
        the weight of the classification term
    beta2 : float, Default 0.1
        the weight of the structural regularization term
    gamma : float, Default 0.1
        the weight used in the initialization of the classifier
    n_iter : integer, Default 100
             the number of iterations 
    lambda1 : float, Default 0.1
            the weight or the constraint on the sparsity term 
    sc_mode : integer in {0, 1, 2, 3, 4, 5}, Default 2
            the type of lasso formulation to use.
            - 0 : the lasso formulation,
            - 1 : minimize the sparsity s.t. small noise,
            - 2 : the basis pursuit denoising formulation,
            - 3 : l_0 version of the lasso,
            - 4 : l_0 version of the case 1,
            - 5 : l_0 version of the basis pursuit denoising.
    pos_coef : bool, Default False
            positivy constrain on the sparse codes 
    pos_dico : bool, Default False
            positivy constrain on the atoms' values 
    rand_init : bool, Default 
        whether or not use a random initialization of the dictionary
    init_iter : int, Default
        number of iterations to perform in initialization per dictionary
     
    Returns
    -------
    self : object
        Returns self. 
    """
    def __init__(self, n_atoms=80, beta1=0.1, beta2=0.1, gamma=0, technic=1,
             lambda1=0.1, sc_mode=2, n_iter=20, lambda2=0, 
             dico_algo='sgd', rand_init=False, init_iter=100,
             init_D=None, weighted_lasso=False, min_weight=10e-3, 
             return_model=False, n_jobs=8, init_W=None, pos_coef=False, 
             pos_dico=False): 
             
        self.name      = 'JDCL' 
        
        self.technic   = technic
        self.beta1     = beta1
        self.beta2     = beta2
        self.gamma     = gamma
         
        self.dico_algo = dico_algo
        self.sc_mode   = sc_mode
        
        self.n_atoms   = n_atoms
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.n_iter    = n_iter
        self.n_jobs    = n_jobs 
        
        self.D         = init_D
        self.model     = None  
        self.return_model= True
        self.l_time    = 0
         
        self.n_cls = -1

        self.rand_init = rand_init
        self.init_iter = init_iter
        
        self.weighted_lasso= weighted_lasso
        self.min_weight    = min_weight 
    
        self.W         = init_W 
        
        self.pos_coef  = pos_coef
        self.pos_dico  = pos_dico
        """
        self.short_print = '%s__%d__%d__%.2f__%.2f__wl_%d__%d__%d__%d__%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.beta1, self.gamma, 
                          self.weighted_lasso, self.rand_init, self.pos_coef,  
                          self.sc_mode, self.lambda1)
        self.full_print= '%s__atoms_%d__iter_%d__beta_%.2f__gamma_%.2f__wl_%d'\
                            +'__rinit_%d__pcoef_%s__mlass_%s__lambda_%.2f__' \
                        %(self.name, self.n_atoms, self.n_iter, self.beta1, self.gamma,
                          self.weighted_lasso, self.rand_init, self.pos_coef,
                          self.sc_mode, self.lambda1)
        """
    
    def __str__(self):
        s = 'JDCL (Based on Zhang2010) --- Optimization \n'
        s+= 'n_atoms:%d __ n_iter:%d __ rand_init:%d \n' %(self.n_atoms,
                                                           self.n_iter,
                                                           self.rand_init)
        s+= 'lambda1:%.2f __ lambda2:%.2f \n' %(self.lambda1, self.lambda2)
        s+= 'beta1:%.2f __ beta2:%.2f \n' %(self.beta1, self.beta2)
        s+= 'dico_algo:%s __ sc_mode:%s \n' %(self.dico_algo, self.sc_mode)   
        s+= 'n_cls:%s __ technic:%s' %(self.n_cls, self.technic)
        return s
     
    def fit(self, data, labels):
                
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        labels : array-like, shape (n_samples,)
            Target vector assosiated with X. 
            
        Returns
        -------
        self : object
            Returns self.
        """ 
        self.n_cls = np.unique(labels).astype('int')
        assert(len(self.n_cls) >= 2)
         
        
        X = [data[labels==i, :] for i in self.n_cls]
        
        print(self)
         
        H = []
        Q = [] if self.technic == 1 else None
 
        if self.D is None:
            self.D = init_sub_dictionaries(X, self.rand_init, self.n_atoms, 
                                           self.lambda1, self.sc_mode, 
                                           self.init_iter)  
            self.D = np.concatenate(self.D, axis=0)
            #self.D = np.array(self.D).transpose(2, 1, 0) 
            #self.D = self.D.reshape((-1, X[0].shape[1])) 
        
        for i in self.n_cls:
            if Q is not None:
                Q = linalg.block_diag(Q, np.ones((self.n_atoms, X[i].shape[0])))
            h = np.zeros((X[i].shape[0], len(self.n_cls),)) 
            h[:, i] = 1    
            H.append(h)
        
        X = np.concatenate(X, axis=0)
        H = np.concatenate(H, axis=0)
        Q = Q.T if Q is not None else Q
        Q = Q[:,1:] if Q is not None else Q
        
        A      = self.decode(X)
        try: 
            pinv   = linalg.pinv2(A)
            K      = np.dot(pinv, Q) if self.technic == 1 else None
            self.W = np.dot(pinv, H) if self.W is None else self.W
        except :
            print("Unexpected error:", sys.exc_info()[0])
            K = np.random.randn(self.D.shape[0], self.D.shape[0])
            K = utils_funcs.norm_cols(K)
            self.W = np.random.randn(self.D.shape[0], len(self.n_cls))
            self.W = utils_funcs.norm_cols(self.W)
        

        if self.technic == 0:
            Dnew = np.hstack((self.D,  np.sqrt(self.beta1)*self.W))
            Xnew = np.hstack((X, np.sqrt(self.beta1)*H))
        else:  
            Dnew = np.concatenate((self.D, np.sqrt(self.beta1)*K, np.sqrt(self.beta2)*self.W), axis=1)
            Xnew = np.concatenate((X, np.sqrt(self.beta1)*Q, np.sqrt(self.beta2)*H), axis=1)
        Dnew =utils_funcs.norm_cols(Dnew)
        
        params = {'n_atoms':self.n_atoms, 'lambda1':self.lambda1, 
                  'n_iter':self.n_iter, 'model':self.model, 
                  'return_model':self.return_model, 'init_D':Dnew.T }
        tic = time.time()
        if self.dico_algo == 'ks':
            Dh, self.model =utils_dl_optim.ksvd(Xnew, **params)
        else: 
            params['mode']     = self.sc_mode
            params['lambda2']  = self.lambda2
            params['pos_coef'] = self.pos_coef
            params['pos_dico'] = self.pos_dico    
            params['n_jobs']   = self.n_jobs    
            Dh, self.model =utils_dl_optim.odl(Xnew.T, **params)
            Dh = Dh.T
    
        self.l_time = time.time() - tic
        
        self.D = Dh[:, :self.D.shape[1]] 
        self.W = Dh[:, -self.W.shape[1]:]        
        
        self.D = utils_funcs.norm_cols(self.D)
        self.W = utils_funcs.norm_cols(self.W)    
    
    def decode(self, X):
        W = None
        if self.weighted_lasso:
            W = []
            for p in range(self.n_atoms):
                w = X - np.tile(self.D[p, :], (X.shape[0],1)).T
                w = np.sqrt((w**2).sum(0))
                W.append(w)
            W = np.array(W, order='FORTRAN')
            W = np.where(W==0, self.w_lasso_min, W)
        
        params = {'sc_mode':self.sc_mode, 'lambda1':self.lambda1, 'W':W, 
                  'lambda2':self.lambda2, 'n_jobs':self.n_jobs,
                  'pos_coef':self.pos_coef}     
        A      = utils_dl_optim.sparse_decode(X.T, self.D.T, **params).T
        return A
    
    def model_conv(self, X, old_D=None):
        e = self.rec_error(X)
        c = None
        if old_D is not None:
            c = np.sqrt(((self.D-old_D)**2).sum())/np.sqrt((old_D**2).sum())
        return [e, c]
    
    def rec_error(self, X):
        Xh = self.reconstruct(X)
        e = 0.5*((X - Xh)**2).sum()
        return e
    
    def reconstruct(self, X):
        A = self.decode(X)
        return np.dot(A, self.D)
        
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.    
        """  
        A = self.decode(X)
        pred = np.dot(A, self.W) 
        return np.argmax(pred, 1)
    
    def predict_proba(self, X):
        A = self.decode(X)
        pred = np.dot(A, self.W)  
        pred =utils_funcs.softmax(pred, 1)
        return pred
    



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################




class SRMC(InputToMap):
    """
    Patch segmentation based on Image super-resolution models.
    
    Parameters
    ----------
    n_atoms : integer, Default 100
              the number of atoms () 
    technic : integer in {0, 1, 2, 3, 4, 5}, Default 2
            the type of lasso formulation to use.
            - 0 : Yang et al.
            - 1 : Yang et al. + classification term,
            - 2 : Zeyde et al.,
            - 3 : Timofte et al. - Grouped,
            - 4 : Timofte et al. - ANR,
            - 5 : Map between representation.
    beta1 : float, Default 0.1
        the weight on the low-res dictionary term
    beta2 : Used if "technic" equals 1, float, Default 0.1
        the weight of the classification term 
    gamma : Used if "technic" equals 1, float, Default 0.1
        the weight used in the initialization of the classifier 
        the threshold value used to remove shared atoms, see the paper
    n_iter : integer, Default 100
             the number of iterations 
    lambda1 : float, Default 0.1
            the weight or the constraint on the sparsity term 
    sc_mode : integer in {0, 1, 2, 3, 4, 5}, Default 2
            the type of lasso formulation to use.
            - 0 : the lasso formulation,
            - 1 : minimize the sparsity s.t. small noise,
            - 2 : the basis pursuit denoising formulation,
            - 3 : l_0 version of the lasso,
            - 4 : l_0 version of the case 1,
            - 5 : l_0 version of the basis pursuit denoising.
    pos_coef : bool, Default False
            positivy constrain on the sparse codes 
    pos_dico : bool, Default False
            positivy constrain on the atoms' values 
    rand_init : bool, Default 
        whether or not use a random initialization of the dictionary
    init_iter : int, Default
        number of iterations to perform in initialization per dictionary
     
    Returns
    -------
    self : object
        Returns self. 
    """ 
    def footprint(self): 
        return '%s__%s__%s__%d__%d__%.2f__%d__' \
                %(self.name, self.technic, self.n_atoms, self.n_iter, 
                          self.sc_mode, self.lambda1, self.proj_tech)
    def full_print(self):
        return '%s__tec_%s__atoms_%s__iter_%d__mlass_%d__lambda_%.2f__prtec_%d__' \
               %(self.name, self.technic, self.n_atoms, self.n_iter,
               self.sc_mode, self.lambda1, self.proj_tech) 
    
    def __init__(self, n_atoms=[80,80], technic=0, beta1=0.1, beta2=0.1, gamma=0,
             lambda1=0.1, sc_mode=2, n_iter=20, lambda2=0, lambda_anr=0.5,
             proj_tech=0,
             dico_algo='on', rand_init=False, init_iter=100,
             init_D=None, weighted_lasso=False, min_weight=10e-3, 
             return_model=False, n_jobs=8, init_Dl=None, init_Dh=None,
             init_W=None, pos_coef=False, pos_dico=False): 
             
        self.name      = 'SRMC' 
        
        self.technic   = 0
        self.beta1     = beta1
        self.beta2     = beta2
        self.gamma     = gamma 
        self.lambda_anr= lambda_anr
        self.proj_tech = proj_tech
         
         
        self.dico_algo = dico_algo
        self.sc_mode   = sc_mode
        
        self.n_atoms   = n_atoms if hasattr(n_atoms, '__getitem__') else 2*[n_atoms] 
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.n_iter    = n_iter
        self.n_jobs    = n_jobs 
         
        self.return_model= True
        self.l_time    = 0
         
        self.rand_init = rand_init
        self.init_iter = init_iter
        
        self.weighted_lasso= weighted_lasso
        self.min_weight    = min_weight 
     
 
        self.model_l   = None
        self.model_h   = None
        self.P         = None 
        self.Dl        = init_Dl
        self.Dh        = init_Dh
        self.W         = init_W
         
        self.pos_coef  = pos_coef
        self.pos_dico  = pos_dico
         
    def __str__(self):
        s= 'SRMC (Image up-scaling) --- Optimization \n'
        s+= 'n_atoms:%s __ n_iter:%d __ rand_init:%d \n' %(self.n_atoms, 
                                                           self.n_iter,
                                                           self.rand_init)
        s+= 'lambda1:%.3f __ lambda2:%.2f \n' %(self.lambda1, self.lambda2)
        s+= 'beta1:%.3f __ beta2:%.3f __ gamma:%.2f \n' %(self.beta1, 
                                                          self.beta2, 
                                                          self.gamma)
        s+= 'dico_algo:%s __ sc_mode:%s \n' %(self.dico_algo, self.sc_mode)   
        s+= 'technic:%s  \n' %(self.technic)   
        return s
    
    def fit(self, data, labels): 
                
        """Fit the model according to the given training data.
        
        "data" must consist of the signals subset and their corresponding maps
        "data" = [matrix of signals,  matrix of maps]
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features) 
            contains the training signal, and X[1] 
        labels : array-like, shape (n_samples, label_dimension)
            contains there classification maps
            
        Returns
        -------
        self : object
            Returns self.
        """   
           
        Xl = data
        Xh = labels
        
        print(self)
        
        if self.Dl is None:
            self.Dl = init_sub_dictionaries([Xl], self.rand_init, self.n_atoms[0],
                                            self.lambda1, self.sc_mode, self.init_iter) 
            self.Dl = self.Dl[0]
        if self.Dh is None:
            self.Dh = init_sub_dictionaries([Xh], self.rand_init, self.n_atoms[1],
                                            self.lambda1, self.sc_mode, self.init_iter) 
            self.Dh = self.Dh[0]
         
         
        if self.technic == 0:
            self.yang_model(Xl, Xh) 
            
        if self.technic == 2:
            self.zeyde_model(Xl, Xh)
        
        if self.technic == 3:
            self.anrglobal_model(Xl, Xh)
            
        if self.technic == 4:
            self.anr_model(Xl, Xh)
        
        if self.technic == 5:
            self.mbr_model(Xl, Xh)
        return self
    
    def mbr_model(self, Xl, Xh):
        # \|Xl - AD\|_F^2 + \|A\|_1 + \|Xh - BC\|_F^2 + \|B - AP\|_F^2 + \|B\|_F^2 + \|P\|_F^2 
        params = {'n_atoms':self.n_atoms[1], 'lambda1':self.beta2, 
                  'lambda2':self.lambda2, 'mode':self.sc_mode, 
                  'init_D':self.Dh.T, 'n_iter':self.n_iter, 'n_jobs':self.n_jobs,
                  'return_model':self.return_model, 'pos_coef':self.pos_coef, 
                  'pos_dico':self.pos_dico, 'model':self.model_h}      
                  
        self.Dh, self.model_h =utils_dl_optim.odl(Xh.T, **params) 
        self.Dh = self.Dh.T

        params = {'lambda1':self.lambda1, 'lambda2':self.lambda2, 'D':self.Dh.T, 
                  'mode':self.sc_mode, 'pos_coef':self.pos_coef, 'n_jobs':self.n_jobs,}   
        Ah     = self.decode(Xh, params)  
           
                  
        if self.P is None:  
            params = {'lambda1':self.lambda1, 'lambda2':self.lambda2, 'D':self.Dl.T, 
                    'mode':self.sc_mode, 'pos_coef':self.pos_coef, 'n_jobs':self.n_jobs,}  
            Al   = self.decode(Xl, params)  
            try:
                Al   = linalg.pinv2(Al)
                P    = np.dot(Al, Ah)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                P = np.random.randn(self.Dl.shape[0], self.Dl.shape[0])
            P = utils_funcs.norm_cols(P)
            Dnew = np.hstack( (self.Dl, np.sqrt(self.beta1)*P) ) 
        else:
            Dnew = np.hstack((self.Dl, np.sqrt(self.beta1)*self.P))
        Xnew = np.hstack((Xl, np.sqrt(self.beta1)*Ah)) 

        params = {'n_atoms':self.n_atoms[0], 'lambda1':self.lambda1, 
                  'lambda2':self.lambda2, 'mode':self.sc_mode, 
                  'init_D':Dnew.T , 'n_iter':self.n_iter, "n_jobs":self.n_jobs,
                  'return_model':self.return_model, 'pos_coef':self.pos_coef, 
                  'pos_dico':self.pos_dico, 'model':self.model_l}   
                  
        Dnew, self.model_l = utils_dl_optim.odl(Xnew.T, **params)  
        Dnew = Dnew.T
        self.Dl = Dnew[:, :self.Dl.shape[1]] 
        self.P  = Dnew[:, self.Dl.shape[1]:]        
        
        self.Dl = utils_funcs.norm_cols(self.Dl)
        self.P  = utils_funcs.norm_cols(self.P)      
    
        
    def anr_model(self, Xl, Xh):
        # Here we assume that each LR atom is associated with a HR atom of the same index.
    
        self.yang_model(Xl, Xh)
        
        nn  = KNN(self.knn)
        nn.fit(self.Dl)
        
        ngs = []
        for j in range(self.n_atoms): 
            ngs.append(list(nn.kneighbors(self.Dl[j, :])))
 
        self.Pjs = []
        for j in range(self.n_atoms):
            inv = linalg.inv(np.dot(self.Dl[ngs[j], :].T, self.Dl[ngs[j], :]) + self.lambda_anr)
            Pj  = np.dot(self.Dh[ngs[j], :], np.dot(inv, self.Dl[ngs[j], :].T))
            self.Pjs.append(Pj)
    
    def anrglobal_model(self, X, Y):
        self.zeyde_model(X, Y)
        #self.yang_model(X, Y)
        # P = D_h (D_l^T D_l + \lambda I)^-1 D_l^T
        inv    = linalg.inv(np.dot(self.Dl.T, self.Dl) + self.lambda_anr*np.identity(self.Dl.shape[1]))
        self.P = np.dot(self.Dh, np.dot(inv, self.Dl.T))
 
        
    def zeyde_model(self, Xl, Xh):
        
        params = {'n_atoms':self.n_atoms[0], 'lambda1':self.lambda1, 
                  'n_iter':self.n_iter, 'model':self.model_l, 
                  'return_model':self.return_model, 'init_D': self.Dl.T }
        tic = time.time()
        if self.dico_algo == 'ks':
            Dh, self.model_h =utils_dl_optim.ksvd(Xl, **params)
        else: 
            params['mode']     = self.sc_mode
            params['lambda2']  = self.lambda2
            params['pos_coef'] = self.pos_coef
            params['pos_dico'] = self.pos_dico
            params['n_jobs']   = self.n_jobs  
            Dh, self.model_l   = utils_dl_optim.odl(Xl.T, **params)  
        Dh = Dh.T
        A      = self.decode(Xl) 
        pinvA  = linalg.pinv2(A) 
        
        self.Dh = np.dot(pinvA, Xh) 
        
        self.l_time = time.time() - tic 
        
    def yang_model(self, Xl, Xh, H=None):
        A = self.decode(Xl)  
        
        if H is not None:
            invA   =  linalg.inv(np.dot(A, A.T) + self.gamma)
            self.W = np.dot(np.dot(H, A.T), invA)     
            Dnew = np.concatenate((np.sqrt(self.beta1)*self.Dl, self.Dh, np.sqrt(self.beta2)*self.W), axis=1)
            Xnew = np.concatenate((np.sqrt(self.beta1)*Xl,  Xh, np.sqrt(self.beta2)*H), axis=1) 
        else:
            Dnew = np.hstack(( np.sqrt(self.beta1)*self.Dl, self.Dh ))
            Xnew = np.hstack(( np.sqrt(self.beta1)*Xl,  Xh  ))     
        Dnew =utils_funcs.norm_cols(Dnew)
        
        params = {'n_atoms':self.n_atoms[0], 'lambda1':self.lambda1, 
                  'n_iter':self.n_iter, 'model':self.model_l, 'n_jobs':self.n_jobs,
                  'return_model':self.return_model, 'init_D': Dnew.T }
        
        tic = time.time()
        if self.dico_algo == 'ks':
            Dh, self.model_l =utils_dl_optim.ksvd(Xnew.T, **params)
        else: 
            params['mode']     = self.sc_mode
            params['lambda2']  = self.lambda2
            params['pos_coef'] = self.pos_coef
            params['pos_dico'] = self.pos_dico  
            Dh,self.model_l   = utils_dl_optim.odl(Xnew,  **params)  
        Dh = Dh.T
        self.Dl = Dh[:, :self.Dl.shape[1]]      
        self.Dh = Dh[:, self.Dh.shape[1]:self.Dh.shape[1]*2]
        self.Dl = utils_funcs.norm_cols(self.Dl) 
        self.Dh = utils_funcs.norm_cols(self.Dh) 
        if H is not None:
            self.W  = Dh[:, self.Dh.shape[1]*2:]
            self.W  = utils_funcs.norm_cols(self.W) 
        
        self.l_time = time.time() - tic  
        
    def decode(self, X, params=None):
        if params is None:
            params = {'D':self.Dl.T, 'sc_mode':self.sc_mode,
                      'lambda1':self.lambda1, 'lambda2':self.lambda2, 
                      'n_jobs':self.n_jobs, 'pos_coef':self.pos_coef}
        A =utils_dl_optim.sparse_decode(X.T, **params).T
        return A
    
    def model_conv(self, Xl, Xh, old_Dl=None, old_Dh=None):
        e  = self.rec_error(Xl, Xh)
        c1 = None
        if old_Dl is not None:
            c1 = np.sqrt(((self.D-old_Dl)**2).sum())/np.sqrt((old_Dl**2).sum())
        c2 = None
        if old_Dl is not None:
            c2 = np.sqrt(((self.D-old_Dh)**2).sum())/np.sqrt((old_Dh**2).sum())
        return [e, c1, c2]
    
    def rec_error(self, Xl, Xh):
        A = self.decode(Xl)
        es = 0.5*((Xl - np.dot(A, self.Dl))**2).sum()
        ec = 0.5*((Xh - np.dot(A, self.Dh))**2).sum()
        return [es, ec]
    
    def predict(self, Xl):
        """
        Output the classification maps of samples in Xl.
        
        Xl should contain signal vector
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features. 
        Returns
        -------
        Y_pred : array, shape (n_samples,n_features)
            Classification maps of samples in X.    
        """  
        if self.technic == 3:
            return np.dot(self.P, Xl)
        
        if self.technic == 4:
            res = []
            app = res.append
            for i in range(Xl.shape[1]):
                j =utils_funcs.knn(Xl[i, :], self.Dl, 1, 'corr')
                app(np.dot(self.Pjs[j], Xl[i, :]))
            return np.array(res).T
        
        A = self.decode(Xl)
        if self.technic == 1:
            res = np.dot(A, self.Dh)
            pre = np.argmax(np.dot(A, self.W), 1)
            res[:,pre==0] = 0
            return res
        
        if self.technic == 5:
            Ah = np.dot(A, self.P) 
            return np.dot(Ah, self.Dh)
        
        return np.dot(A, self.Dh)