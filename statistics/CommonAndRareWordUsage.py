#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:26:00 2018

@author: c.senik
"""



import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pprint import pprint
import itertools

class WordUsage(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordUsage, self).__init__(path, self.__class__.__name__, suffix)
        
        n_rare=3000   #for train, vocab of 9755 words
        n_common=100
        
        n_rare=40   #for greedy, vocab of 241 words
        n_common=10
        
        max_rare=6
        max_common=20

        stopwords = [" ","the","a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]
        questions = []
        status_count = collections.Counter()
        status_list = []

        for i, game in enumerate(games):
            status_count[game.status] += 1
            status_list.append(game.status)
            questions.append(game.questions)
        questions = list(itertools.chain(*questions))
        tot_words=0
        #word_list = []
        word_counter = collections.Counter()
        for q in questions:
            q = re.sub('[?]', '', q)
            words = re.findall(r'\w+', q)
            for w in words:
                if w.lower() not in stopwords:
                    tot_words+=1
                    word_counter[w.lower()] += 1
                
        word_list=[[word_counter[w],w] for w in word_counter]
        word_list.sort()
        #print(word_list)
        print(len(word_list))
        
        rare_words=[w[1] for w in word_list[:n_rare]]
        common_words=[w[1] for w in word_list[-n_common:]]
        
        
        
        #word_list = list(itertools.chain(*word_list))

       

        ###########################

        
        # make these smaller to increase the resolution
        dx, dy = 1, 1
        
        # generate 2 2d grids for the x & y bounds  x for common, y for rare
        y,x = np.mgrid[slice(0, max_rare + dy, dy),
                        slice(0, max_common + dx, dx)]
        
        
        
        z=np.ndarray(shape=(max_common,max_rare), dtype=float, order='F')
        
        # let's build in Z the average of success for given numbers of rare an common words
        
        Z=np.array([[[0,0] for i in range (max_rare) ] for j in range(max_common)])
        
        for ind,g in enumerate(games):
            succ=1.0*int((status_list[ind]=="success"))
            qs=g.questions
            nb_rare=0
            nb_common=0
            for q in qs:
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)
                for w in words:
                    if w.lower() in rare_words:
                        nb_rare+=1
                    if w.lower() in common_words:
                        nb_common+=1
            if (nb_rare<max_rare and nb_common<max_common):
                [average,n_q]=Z[nb_common][nb_rare]
                Z[nb_common][nb_rare]=[(1.0*average*n_q + succ)/(n_q+1),n_q+1]
        for i in range(max_common):
            for j in range(max_rare):
                z[i][j]=Z[i][j][0]
                
                        
                        
        z=np.transpose(z)       
            
            
     
        #z = z[:-1, :-1]
        print(np.shape(z))
        print(np.shape(x))
        print(np.shape(y))
        levels = MaxNLocator(nbins=10).tick_values(z.min(), z.max())
        
        
        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.
        cmap = plt.get_cmap('PiYG')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        
        fig,  ax1 = plt.subplots(nrows=1)
        
        #im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        #fig.colorbar(im, ax=ax0)
        #ax0.set_title('pcolormesh with levels')
        #ax0.set_xlabel("Number of common words in the dialogue", {'size':'6'})
        #ax0.set_ylabel("Number of rare words in the dialogue", {'size':'6'})
        
        
        # contours are *point* based plots, so convert our bound into point
        # centers
        cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                          y[:-1, :-1] + dy/2., z, levels=levels,
                          cmap=cmap)
        fig.colorbar(cf, ax=ax1)
        ax1.set_title('contourf with levels')
        ax1.set_xlabel("Number of common words in the dialogue", {'size':'6'})
        ax1.set_ylabel("Number of rare words in the dialogue", {'size':'6'})
        
        # adjust spacing between subplots so `ax1` title and `ax0` tick labels
        # don't overlap
        fig.tight_layout()
        
        plt.show()



