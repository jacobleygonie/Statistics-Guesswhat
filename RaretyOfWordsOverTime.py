#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:14:24 2018

@author: c.senik
"""

import collections

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import sys
import json
from pprint import pprint
import itertools


from guesswhat.statistics.abstract_plotter import *




class RaretyVSTime(AbstractPlotter):
      
    def __init__(self, path, games, logger, suffix):
        
        super(RaretyVSTime, self).__init__(path, self.__class__.__name__, suffix)
        stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]
        questions = []

        for i, game in enumerate(games):
            
            questions.append(game.questions)
        questions = list(itertools.chain(*questions))
        max_occ=0
        word_list = []
        word_counter = collections.Counter()
        for q in questions:
            q = re.sub('[?]', '', q)
            words = re.findall(r'\w+', q)
            for w in words:
                if w not in stopwords:
                    
                    word_list.append(words)
                    word_counter[w.lower()] += 1

        word_list = list(itertools.chain(*word_list))

        for w in word_counter:
            if word_counter[w]>max_occ:
                max_occ=word_counter[w]
        
        rareties=collections.Counter()
        for w in word_counter:
            rareties[w]=float(word_counter[w])
        
        #in the following we will consider only the dialogues of size 5 and evaluate the evolution of words rarety
        
        #put len(g.questions)==5 in general except for final7 put 7
        questions_five = [g for g in games if len(g.questions)==5]
        print("There are"+str(len(questions_five)) + "dialogues of lenght five")
        questionsVSrarety=[]
        for g in questions_five: 
            q=g.questions
            for i in range(5): #put 5 usually, 7 for final7
                cur=q[i]
                cur = re.sub('[?]', '', cur)
                words = re.findall(r'\w+', cur)
                for w in words:
                    if w.lower() in rareties:
                        questionsVSrarety.append([i,rareties[w.lower()]])
                
        
        questionsVSrarety=np.array(questionsVSrarety)
        
        counter = collections.defaultdict(list)
        for k, val in questionsVSrarety:
            counter[k] += [val]
        
        #put arr = np.zeros( [4, 5]) normally, 7 for final7
        arr = np.zeros( [4, 5])
        for k, val in counter.items():
            if len(val) > 0:
                print(k)
                arr[0,int(k)] = k+1
                arr[1,int(k)] = np.mean(val)

                # Std
                arr[2, int(k)] = np.std(val)

                # confidence interval 95%
                arr[3,int(k)] = 1.95*np.std(val)/np.sqrt(len(val))
        
       
        plt.plot(arr[0,:],arr[1,:], label="word frequency for dialogues of lenght 5, in function the index of the question", marker="o")
        #plt.fill_between(x=arr[0,:], y1=arr[1,:]-arr[2,:], y2=arr[1,:]+arr[2,:], alpha=0.2)
        #x = np.linspace(0, 5, 10)
        #sns.regplot    (x=x, y=np.log2(x), order=6, scatter=False, label="y = log2(x)", line_kws={'linestyle':'--'})
        #f = sns.regplot(x=x, y=x         , order=1, scatter=False, label="y = x"      , line_kws={'linestyle':'--'})

        
        #put usually 6, 8 for final7
        plt.xlim(0,8)
        plt.xlabel("Index of question", {'size':'14'})
        plt.ylabel("frequency of words", {'size':'14'})
                        
            
                        
                        
                
        
        
        
        
                
        
        
