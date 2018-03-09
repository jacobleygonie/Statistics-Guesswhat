#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:26:31 2018

@author: clairelasserre
"""


import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import matplotlib.pyplot as plt

import collections

class RatioYesVsTime(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(RatioYesVsTime, self).__init__(path, self.__class__.__name__, suffix)

        couples_t_ratioyes_success = [] #contains triplets triplet =(time, current % of yes, current yes+no)
        couples_t_ratioyes_failure = []
        
        t_max_success=0
        t_max_failure=0

        for game in games:
            if (game.status=="success"):
                no_question = len(game.answers)
                if (no_question>t_max_success):
                    t_max_success = no_question
            if (game.status=="failure"):
                no_question = len(game.answers)
                if (no_question>t_max_failure):
                    t_max_failure = no_question
        couples_t_ratioyes_success = [(i,0,0) for i in range (0,t_max_success)]
        couples_t_ratioyes_failure = [(i,0,0) for i in range (0,t_max_failure)]
        
        for game in games:
            if (game.status=="success"):
                no_answers = len(game.answers)
                for t in range(no_answers):
                    a = game.answers[t]
                    a = a.lower()
                    if a == "yes": #then we actualize the tables
                    
                        old_percent= couples_t_ratioyes_success[t][1]
                        old_sum = couples_t_ratioyes_success[t][2]
                        couples_t_ratioyes_success[t] = (t, (old_percent*(old_sum)+1)/(old_sum+1),old_sum+1)
                    if a== "no":
                        old_percent= couples_t_ratioyes_success[t][1]
                        old_sum = couples_t_ratioyes_success[t][2]
                        couples_t_ratioyes_success[t] = (t, (old_percent*(old_sum))/(old_sum+1),old_sum+1)
            if (game.status=="failure"):
                no_answers = len(game.answers)
                for t in range(no_answers):
                    a = game.answers[t]
                    a = a.lower()
                    if a == "yes": #then we actualize the tables
                    
                        old_percent= couples_t_ratioyes_failure[t][1]
                        old_sum = couples_t_ratioyes_failure[t][2]
                        couples_t_ratioyes_failure[t] = (t, (old_percent*(old_sum)+1)/(old_sum+1),old_sum+1)
                    if a== "no":
                        old_percent= couples_t_ratioyes_failure[t][1]
                        old_sum = couples_t_ratioyes_failure[t][2]
                        couples_t_ratioyes_failure[t] = (t, (old_percent*(old_sum))/(old_sum+1),old_sum+1)
                
        
        couples_t_ratioyes_success_new = [(couples_t_ratioyes_success[t][0],couples_t_ratioyes_success[t][1]) for t in range(0,t_max_success)]
        couples_t_ratioyes_failure_new = [(couples_t_ratioyes_failure[t][0],couples_t_ratioyes_failure[t][1]) for t in range(0,t_max_failure)]
        
        couples_t_ratioyes_success_new = np.array(couples_t_ratioyes_success_new)
        couples_t_ratioyes_failure_new = np.array(couples_t_ratioyes_failure_new)
        
        
        
        sns.set(style="white")
        sns.regplot(x=couples_t_ratioyes_success_new[:, 0], y=couples_t_ratioyes_success_new[:, 1], x_ci=None,   label="Succes", marker="o", line_kws={'linestyle':'-'})
        f=sns.regplot(x=couples_t_ratioyes_failure_new[:, 0], y=couples_t_ratioyes_failure_new[:, 1], x_ci=None, label="Failure", marker="o", line_kws={'linestyle':'-'})

        
        f.legend(loc="best", fontsize='x-large')
        f.set_xlim(0,max(t_max_success, t_max_failure))
        #f.set_ylim(0,20)
        f.set_xlabel("Time", {'size':'14'})
        f.set_ylabel("% of yes (in mean)", {'size':'14'})