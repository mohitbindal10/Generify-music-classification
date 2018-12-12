#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:11:45 2017

@author: mohitbindal
"""

import os
import glob
import sys
import numpy as np
import scipy
import scipy.io.wavfile
from python_speech_features import mfcc
import matplotlib
import matplotlib.pyplot
from matplotlib.pyplot import specgram
from utils import ENGLISH_GENRE_DIR, CHART_DIR, ENGLISH_GENRE_LIST
genre_list=ENGLISH_GENRE_LIST
GENRE_DIR=ENGLISH_GENRE_DIR
def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print ("Written %s", data_fn)

def create_ceps(fn):
    """
        Creates the MFCC features. 
    """    
    sample_rate, X = scipy.io.wavfile.read(fn)
    #print("samplerate=",sample_rate)
    #print("X=",X,len(X),X.shape)
    Y=X.ravel()
    #print("Y=",Y,len(Y),Y.shape)    
    #X[X==0]=1
    #specgram(X, Fs=sample_rate,xextent=(0,30))
    #matplotlib.pyplot.show
    ceps = mfcc(Y)
    mspec = mfcc(Y)
    spec = mfcc(Y)
    #print("ceps=",ceps)
    #print("mspec=",mspec)

    #print("spec=",spec)

    write_ceps(ceps, fn)


def read_ceps(genre_list, base_dir=GENRE_DIR):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            #filename,extention=os.path.splitext(fn)
            #f1,f2,f3=filename.split('.')
            num_ceps = len(ceps)
            X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
           
            y.append(label)
    return np.array(X), np.array(y)


def create_ceps_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    sample_rate, X = scipy.io.wavfile.read(fn)
    X[X==0]=1
    np.nan_to_num(X)
    X=X.ravel()
    ceps, mspec, spec = mfcc(X),mfcc(X),mfcc(X)
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print ("Written ", data_fn)
    return data_fn


def read_ceps_test(test_file):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    ceps = np.load(test_file)
    #filename,extention=os.path.splitext(test_file)
    #f1,f2,f3=filename.split('.')
    num_ceps = len(ceps)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    label=10
                
    #y.append(label)
    return np.array(X)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection(set(ENGLISH_GENRE_LIST) ))
        break
    print ("Working with these genres --> ", traverse)
    print ("Starting ceps generation")     
    for subdir, dirs, files in os.walk(GENRE_DIR):
        for file in files:
            path = subdir+'/'+file
            if path.endswith(".wav"):
                print(path)
                tmp = subdir[subdir.rfind('/',0)+1:]
                if tmp in traverse:
                    create_ceps(path)
    
    stop = timeit.default_timer()
    print ("Total ceps generation and feature writing time (s) = ", (stop - start) )
