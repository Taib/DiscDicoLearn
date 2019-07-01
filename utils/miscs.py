# -*- coding: utf-8 -*-
"""Usefull methods for plotting, reading, etc."""
 
# Author: Taibou Birgui Sekou <taibou.birgui_sekou@insa-cvl.fr> 

from __future__ import print_function
import sys

import glob
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import itertools

from matplotlib.widgets import Slider 
from skimage import  filters
 

def range_with_status(beg, end, text):
    """ iterate from 0 to total and show progress in console """
    done = '#' * (beg + 1)
    todo = '-' * (end - beg - 1)
    s = '{0}{1}'.format(done + todo, text) 
    if not todo:
        s += '\n'
    if beg > 0:
        s = '\r' + s
    print(s, end='\r')
    sys.stdout.flush()


def seg_metrics(pred, ann):    
    """ Compute the spec, sens and acc between two ndarrays.
    """    
    acc = [((pred==j)*(ann==j)).sum(dtype='float')/(ann==j).sum() for j in np.unique(ann)] 
    acc.append((pred==ann).sum()/float(ann.size))  
    acc.append(f1_score(pred.flatten(), ann.flatten())) 
    return acc
 
def file_names(path, ext=None, beg=None):
    assert(ext is not None or beg is not None)
    files = glob.glob(path+'*.'+ext) if ext is not None else glob.glob(path+beg+'*')
    files = np.array([k.split('/')[-1] for k in files]) 
    return files
    
def plot_patches(P, p_shape, colored=False, labels=None, title='Patches'):
    plt.figure(figsize=(4.2, 4))
    l, c = int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0])) +1
    s    = np.prod(p_shape)
    for i, comp in enumerate(P):
        if(i+1 > (l*c) ):
            break
        plt.subplot(l, c, i+1)
        if colored: 
            chan = [comp[ch*s:ch*s + s].reshape(p_shape) for ch in range(3)]
            chan = np.array(chan).transpose(1,2,0)
            plt.imshow(chan)
            #plt.imshow(comp.reshape((p_shape[0], p_shape[1], -1)), interpolation="nearest" )
        else:
            
            col = plt.cm.Reds if labels is not None and labels[i]==1 else plt.cm.gray
            plt.imshow(comp.reshape(p_shape), cmap=col, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
        
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    

def plot_codes(P, labels=None, title='Codes'):
    plt.figure( )
    l, c = P.shape[0]/2, 2
    for i, comp in enumerate(P):
        if(i+1 > (l*c) ):
            break
        plt.subplot(l, c, i+1)
        col = 'r' if labels is not None and labels[i]==1 else 'b'
        plt.plot(range(len(comp)), comp, col)
        plt.xticks(())
        plt.yticks(())
        
    plt.suptitle(title, fontsize=16)


def plot_tep(dat):
    import matplotlib.pylab as plt
    from matplotlib.widgets import Slider, Button
    
    fig, ax = plt.subplots()
    plt.subplots_adjust( bottom=0.25)  
    plt.imshow(dat[10, 10], plt.cm.gray) 
    
    axcolor = 'lightgoldenrodyellow'
    axZ = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    axT = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    sZ = Slider(axZ, 'Z', 1, dat.shape[1], valinit=0)
    sT = Slider(axT, 'T', 1, dat.shape[0], valinit=0)
    
    
    def update(val):
        T = int(sT.val)
        Z = int(sZ.val) 
        ax.imshow(dat[T, Z], plt.cm.gray) 
    sZ.on_changed(update)
    sT.on_changed(update)
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    
    def reset(event):
        sZ.reset()
        sT.reset()
    button.on_clicked(reset)
     
    plt.show()


def save_confusion_matrix(pred, gt, plot_saves,
                          title='Confusion matrix'):
    # from sklearn example 
                          
    n_cl = len(np.unique(gt))
    cm =  confusion_matrix(gt, pred)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title + ' -- acc=%.4f' %(pred == gt).mean() )
    plt.colorbar()
    tick_marks = np.arange(n_cl)
    plt.xticks(tick_marks, range(n_cl), rotation=45)
    plt.yticks(tick_marks, range(n_cl))

    # normalization
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(plot_saves + ('_%.3f.png' %(pred == gt).mean() ) , bbox_inches='tight')
    return cm


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment')
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)


class SlideImgThres():
    def __init__(self, image, gt=None):
        assert(image.ndim == 2)
        self.img = image
        self.gt  = gt
        self.inc = 1
        
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust( bottom=0.25)  
        plt.imshow(image, plt.cm.gray) 
        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03],
                                          axisbg='yellow')
                                          
        self.slider = DiscreteSlider(self.sliderax, 'Value', 0, 500, 
                                     increment=self.inc, valinit=self.inc)
        self.slider.on_changed(self.update)
        
        plt.show()

    def update(self, value):
        self.thres = np.where(self.img > filters.threshold_otsu(self.img , value), 1, 0) 
        if self.gt is not None:
            self.fig.suptitle('{0}'.format(seg_metrics(self.thres, self.gt)))
        self.ax.imshow(self.thres)  
        self.fig.canvas.draw()
 