#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:34:55 2018

@author: clairelasserre
"""

#select the most common category and gives a pie chart with the proportion of dialogues that use it as a word, 
# the proportion of dialogues that don't use it as a word, 
#the proportion of dialogues for which there is no category with more than one object in the image

import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import matplotlib.pyplot as plt
import collections
from collections import Counter 

import re
class SeekingCategory(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SeekingCategory, self).__init__(path, self.__class__.__name__, suffix)

        main_category_used_success=0
        main_category_used_failure=0
        main_category_unused_success=0
        main_category_unused_failure=0
        no_imp_category=0

        for game in games:
            categories=[]
            for obj in game.objects : 
                categories.append(obj.category.lower())
            main_cat = Counter(categories).most_common()[0][0]
            nb_main_cat = Counter(categories).most_common()[0][1]
            
            allwords=[]
            no_questions = len(game.questions)
            for t in range(no_questions):
                q = game.questions[t]
                words = re.findall(r'\w+', q)
                for w in words:
                    allwords.append(w.lower())
            if (nb_main_cat>1):
                if(main_cat in allwords):
                    if(game.status=="success"):
                        main_category_used_success+=1
                    if(game.status=="failure"):
                        main_category_used_failure+=1
                else:
                    if(game.status=="success"):
                        main_category_unused_success+=1
                    if(game.status=="failure"):
                        main_category_unused_failure+=1
            else:
                no_imp_category+=1
            
            """for the train: 
                main_category_used_success = 50082
                main_category_used_failure = 6945
                main_category_unused_success = 33018
                main_category_unused_failure = 4316
                no_imp_category = 12660
            """
            
            
        
        labels = 'Main category used', 'Main category unused', "No category with more than 1 object"
        colors = ['yellowgreen', 'gold', 'lightskyblue']
        plt.title("Is the category most represented in the image is used as a word in the dialogue?")
        plt.pie([main_category_used_success+main_category_used_failure,main_category_unused_success+main_category_unused_failure,no_imp_category], labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)

        plt.axis('equal')
        
        
        """  #to distinguish failure and success
        labels = 'main_category_used_success', 'main_category_used_failure', 'main_category_unused_success', 'main_category_unused_failure',"no_imp_category"
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red']
        plt.title("Is the category most represented in the image is used as a word in the dialogue?")
        plt.pie([main_category_used_success,main_category_used_failure,main_category_unused_success,main_category_unused_failure,no_imp_category], labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)

        plt.axis('equal')
        """
        

        
