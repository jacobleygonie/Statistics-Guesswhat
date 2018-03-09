#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:51:59 2018

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

from guesswhat.statistics.abstract_plotter import *

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

class LessUsedWords(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(LessUsedWords, self).__init__(path, self.__class__.__name__, suffix)

        questions = []

        for game in games:
            questions.append(game.questions)
        questions = list(itertools.chain(*questions))


        # split questions into words
        word_list = []
        word_counter = collections.Counter()
        for q in questions:
            q = re.sub('[?]', '', q)
            words = re.findall(r'\w+', q)
            word_list.append(words)

            for w in words:
                if w.lower() not in stopwords:
                    word_counter[w.lower()] += 1


        word_list = list(itertools.chain(*word_list))
        #pprint(word_counter)

        word_list=[[word_counter[w],w] for w in word_counter]
        word_list.sort()
        #print(word_list)
        #print(len(word_list))
        
        n_rare=100  #for train, vocab of 9755 words
        #n_common=30
        rare_words=[w[1] for w in word_list[:n_rare]]
        #common_words=[w[1] for w in word_list[-n_common:]]
        occ_rare_words=[w[0] for w in word_list[:n_rare]]
        #columns = ['Word', 'Number of Occurences']
        data = np.array([rare_words, occ_rare_words]).transpose()
        #print(data)
        
        # Now examine first questions to see if their usage of rare words is related to an object in the image.
        correlation_rword_image={}
        
        for game in games:
            for q in game.questions:
                #print(q)
                q = re.sub('[?]', '', q)
                words = re.findall(r'\w+', q)
                for w in words:
                    if w.lower() in rare_words:
                        is_in_img=False
                        for o in game.objects:
                            c=o.category
                            c=c.lower()
                            if((w.lower() in c) or (c in w.lower())):
                                is_in_img=True
                        is_in_img=int(is_in_img)       
                        if isinstance(correlation_rword_image.get(w.lower()),list):
                            correlation_rword_image[w.lower()][is_in_img]+=1
                        else:
                            correlation_rword_image[w.lower()]=[0,0]
                            correlation_rword_image[w.lower()][is_in_img]+=1
        words= [w for w in correlation_rword_image]
        corr=[]
        fail=[]
        for w in words:
            corr.append(correlation_rword_image[w][1])
            fail.append(correlation_rword_image[w][0])
        
        corr=np.array(corr)
        fail=np.array(fail)         
        
        x=np.arange(len(words))
        
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        ax0, ax1= axes.flatten()
        ax0.tick_params(labelsize=4,rotation=90)
        ax1.tick_params(labelsize=4,rotation=90)
        plt.setp(axes, xticks=x, xticklabels=words)

        
        ax0.bar(x, corr)
        #ax0.xticks(x, words)
        #ax0.legend(prop={'size': 10})
        #ax0.set_title('Presence of rare words in image')
        
        
        ax1.bar(x, fail)
        #ax1.xticks(x, words)
        #ax1.legend(prop={'size': 2})
        #ax1.set_title('Absence of rare words in image',fontsize="xx-small")
            
        #fig.tight_layout()
        plt.show()
        
        """df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Word').sum()
       # df = df.div(df.sum(axis=1), axis=0)
        df = df.sort_values(by='Number of Occurences')
        df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(14,6), color=["g", "b"])
        #df.hist(column='Word', by="Number of Occurences")
        
        sns.set(style="whitegrid")

        plt.xlabel("Rare Words", {'size':'10'})
        plt.ylabel("Occurences in Dialogues", {'size':'10'})"""
        
        
        

