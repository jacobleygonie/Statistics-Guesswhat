#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:24:46 2018

@author: clairelasserre
"""


import json
from pprint import pprint
import itertools
import collections

import wordcloud as wc

import re
import sys

import seaborn as sns
import numpy as np
from guesswhat.statistics.abstract_plotter import *

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

class WordUse(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordUse, self).__init__(path, self.__class__.__name__, suffix)
        
        general_words = ['on', "in", 'of', 'to', "with", "by", "at", "or", "and", "from","top", "left", "right", "side", "next", "front", "middle", "foreground", "bottom", "background",
                       "near", "behind", "back", "at", "row", "far", "whole", "closest",'one', "two", "three", "four", "five", "six", "first", "second", "third", "half", 'visible', "made", "part", "piece", 
                       "all","most","wearing", "have", "can", "holding", "sitting", "building", "standing", "see","green",'blue', 'brown', "red", 'white', "black", "yellow", "color", "orange", "pink"]
        specific_words = ['people', 'person', "he", "she", "human", "man", "woman", "guy", 'alive', "girl", "boy", "head", 'animal',"hand","table", 'car', "food", "plate", "shirt", "something", "thing", "object",
                   "light", "hat", "tree", "bag", "book", "sign", "bottle", "glass", "bus", "wall", "vehicle",
                   "chair", "dog", "cat", "windows", "boat", "item", "shelf", "horse", "furniture", "water", "camera", "bike",
                   "train", "window", "bowl", "plant", "ball", "cup"]
        t_max=0 
        for game in games:
            no_question = len(game.answers)
            if (no_question>t_max):
                t_max = no_question
        if(t_max>30):
            t_max=29
        ratio_t_general = [(i,0,0) for i in range (0,t_max)] #dialogue length, ratio of general, general+specific
    
        
        for game in games:
            no_questions = len(game.questions)
            if (no_questions<30):
                general = 0
                specific =0
                for t in range(no_questions):
                    q = game.questions[t]
                    words = re.findall(r'\w+', q)
                    for w in words:
                        w=w.lower()
                        if (w in general_words):
                            general+=1
                        if (w in specific_words):
                            specific+=1
                if (general!=0 or specific!=0):
                    old_percent= ratio_t_general[no_questions-1][1]
                    old_sum = ratio_t_general[no_questions-1][2]
                    ratio_t_general[no_questions-1] = (no_questions, (old_percent*(old_sum)+general)/(old_sum+general+specific),old_sum+general+specific)
                        
        ratio_t_general = np.array([(ratio_t_general[t][0],ratio_t_general[t][1]) for t in range(0,t_max)])

        sns.set(style="white")
        plt.plot(ratio_t_general[:, 0], ratio_t_general[:, 1])
        #f=sns.regplot(x=couples_t_ratioyes_failure_new[:, 0], y=couples_t_ratioyes_failure_new[:, 1], x_ci=None, label="Failure", marker="o", line_kws={'linestyle':'-'})

        
        #f.legend(loc="best", fontsize='x-large')
        #plt.xlim(1,t_max)
        #f.set_ylim(0,20)
        plt.xlabel("Dialogue Length", {'size':'14'})
        plt.ylabel("% of general words ", {'size':'14'})
        
        
        
        
        """ #first idea  %of general with time but it is non sense (no correlation)-> %of general with dialogue length
        t_max=0

        for game in games:
            no_question = len(game.answers)
            if (no_question>t_max):
                t_max = no_question
        ratio_t_general = [(i,0,0) for i in range (0,t_max)] #time, ratio of general, general+specific
        for game in games:
            no_questions = len(game.questions)
            for t in range(no_questions):
                q = game.questions[t]
                words = re.findall(r'\w+', q)
                for w in words:
                    if (w in general):
                        old_ratiogeneral = ratio_t_general[t][1]
                        old_tot = ratio_t_general[t][1]
                        ratio_t_general[t] = (t,(old_ratiogeneral*old_tot+1)/(old_tot+1),old_tot+1)
                    if (w in specific):
                        old_ratiogeneral = ratio_t_general[t][1]
                        old_tot = ratio_t_general[t][1]
                        ratio_t_general[t] = (t,(old_ratiogeneral*old_tot)/(old_tot+1),old_tot+1)
                        
        ratio_t_general = np.array([(ratio_t_general[t][0],ratio_t_general[t][1]) for t in range(0,t_max)])

        sns.set(style="white")
        f=sns.regplot(x=ratio_t_general[:, 0], y=ratio_t_general[:, 1], x_ci=None,    marker="o", line_kws={'linestyle':'-'})
        #f=sns.regplot(x=couples_t_ratioyes_failure_new[:, 0], y=couples_t_ratioyes_failure_new[:, 1], x_ci=None, label="Failure", marker="o", line_kws={'linestyle':'-'})

        
        #f.legend(loc="best", fontsize='x-large')
        f.set_xlim(0,t_max)
        #f.set_ylim(0,20)
        f.set_xlabel("Time", {'size':'14'})
        f.set_ylabel("% of general words ", {'size':'14'})
        
        """