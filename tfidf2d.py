#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:52:27 2018

@author: c.senik
"""
import argparse

import logging
from logging.handlers import RotatingFileHandler

import matplotlib
matplotlib.use('Agg')

from guesswhat.data_provider.guesswhat_dataset import Dataset

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
from sklearn.preprocessing import Normalizer

from guesswhat.statistics.abstract_plotter import *
import numpy as np
import mpld3

def create_logger(save_path, name):

    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path + '/' + name + '.stats.log', 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]
K=5   #kmeans parameter
DRed=2

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Plotter options!')

    parser.add_argument("-data_dir", type=str, help="Directory with data", required=True)
    parser.add_argument("-out_dir", type=str, help="Output directory", required=True)
    parser.add_argument("-name", type=str, help="Output directory", required=True)
    parser.add_argument("-normalize", type=bool, help="normalize word representation", required=True)
    parser.add_argument("-ignore_incomplete", type=bool, default=True, help="Ignore incomplete games in the dataset")

    args = parser.parse_args()
      
    
    dataset = Dataset(args.data_dir, args.name)
    games = [g for g in dataset.games if g.status == "success" or g.status == "failure"]

    N=len(games)
   
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
    if args.normalize:
        data= Normalizer().fit_transform(np.array(data))
    pca = PCA(n_components=DRed)
    X=pca.fit_transform(data)

    kmeans = KMeans(n_clusters=K, random_state=0,init='k-means++').fit(X) 
    clusters=kmeans.labels_
    
 
    #groups = ("coffee", "tea", "water") 
 
    # Create plot
    """fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
 
    for dat, color in zip(data, colors):
        x, y = dat
        ax.scatter(x, y, c=color, edgecolors='none', s=30,alpha=0.3, cmap=plt.cm.jet)
    
    ax.grid(color='white', linestyle='solid')
 
    """
    #fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
    

    """scatter = ax.scatter(np.random.normal(size=N),
                 np.random.normal(size=N),
                 c=np.random.random(size=N),
                 s=1000 * np.random.random(size=N),
                 alpha=0.3,
                 cmap=plt.cm.jet)
    

    ax.set_title("Scatter Plot (with tooltips!)", size=20)"""
    colors = ["red", "green", "blue","purple","yellow","brown","black","pink","grey"]
    colors=colors[:K]
    data = [[] for i in range (K)]
    print(tot_words)
    x=[]
    y=[]
    col=[]
    size=[]
    labels=word_list
    for i in range (tot_words):
        k=clusters[i]
        x.append(X[i][0])
        y.append(X[i][1])
        col.append(colors[k])
        size.append(0.25*word_counter[word_list[i]])
   
    
    #fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
    fig = plt.figure(figsize=(15, 10)) # 8 inches wide by 6 inches tall
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(np.array(x),
                 np.array(y),
                 c=col,
                 s=size,
                 alpha=0.3,
                 cmap=plt.cm.jet)
    ax.grid(color='white', linestyle='solid')

    ax.set_title('Words Clustering', size=20)
    #print("here we are")
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)
    #print("just before saving")
    
    mpld3.show()
    #mpld3.save_html(fig=fig,fileobj=args.out_dir+"/"+args.name+".tfidf2")
         
         
        
        
        

