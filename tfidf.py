#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:52:27 2018

@author: c.senik
"""


import json
from pprint import pprint
import itertools
import collections

import wordcloud as wc

import re
import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from guesswhat.statistics.abstract_plotter import *

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

class tfidf(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(tfidf, self).__init__(path, self.__class__.__name__, suffix)

        N=len(games)
        K=5   #kmeans parameter
        DRed=2
        data=np.zeros((5, 5))
        
        questions = []
        for game in games:
            questions.append(game.questions)
        questions = list(itertools.chain(*questions))

        # Do the tfidf
        
        # split questions into words
        word_list=[]
        index_word={}
        word_counter = collections.Counter()
        tot_words=0
        for q in questions:
            q = re.sub('[?]', '', q)
            words = re.findall(r'\w+', q)
            for w in words:
                if w.lower() not in stopwords:
                    word_counter[w.lower()] += 1
                    if(word_counter[w.lower()]==1):
                        tot_words+=1
                        word_list.append(w.lower())
                        index_word[w.lower()]=len(word_list)-1
                        
        data=np.zeros((tot_words, N))
        for ind,game in enumerate(games):
            qs=game.questions
            for q in qs:
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)
                for w in words:
                    if w.lower() not in stopwords:
                        index=index_word[w.lower()]
                        data[index][ind]=1
        
        pca = PCA(n_components=DRed)
        X=pca.fit_transform(data)

        kmeans = KMeans(n_clusters=K, random_state=0,init='k-means++').fit(X) 
        clusters=kmeans.labels_
        
        data = [[] for i in range (K)]
        print(tot_words)
        for i in range(K):
            data[i]=[[X[l][0] for l in range(tot_words) if (clusters[l]==i)],[X[l][1] for l in range(tot_words) if (clusters[l]==i)]]
        colors = ["red", "green", "blue","purple","yellow","brown","black","pink","grey"]
        colors=colors[:K]
        #groups = ("coffee", "tea", "water") 
 
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
 
        for dat, color in zip(data, colors):
            x, y = dat
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30)
 
        plt.title('Words Clustering')
        plt.legend(loc=2)
        plt.show()
         
         
        
        
        

