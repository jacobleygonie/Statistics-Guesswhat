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

class MostUsedWords(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(MostUsedWords, self).__init__(path, self.__class__.__name__, suffix)

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
        print(len(word_list))
        
        n_rare=100   #for train, vocab of 9755 words
        n_common=30
        rare_words=[w[1] for w in word_list[:n_rare]]
        common_words=[w[1] for w in word_list[-n_common:]]
        occ_common_words=[w[0] for w in word_list[-n_common:]]
        columns = ['Word', 'Number of Occurences']
        data = np.array([common_words, occ_common_words]).transpose()
        
        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Word').sum()
        #df = df.div(df.sum(axis=1), axis=0)
        df = df.sort_values(by='Number of Occurences')
        df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(14,6), color=["g", "b"])
        sns.set(style="whitegrid")

        plt.xlabel("Common Words", {'size':'14'})
        plt.ylabel("Occurences in Dialogues", {'size':'14'})
        
        

