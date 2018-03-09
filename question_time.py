#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:30:13 2018

@author: clairelasserre
"""

import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import matplotlib.pyplot as plt

import collections

class QuestionVsTime(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(QuestionVsTime, self).__init__(path, self.__class__.__name__, suffix)

        couples_t_question_success = [] #contains triplets triplet =(time, current average of length of questions (if!=null),number of questions)
        couples_t_question_failure = []
        
        t_max_success=0
        t_max_failure=0
        game_suc=0
        game_fail=0
        for game in games:
            if (game.status=="success"):
                game_suc+=1
                no_question = len(game.questions)
                if (no_question>t_max_success):
                    t_max_success = no_question
            if (game.status=="failure"):
                game_fail+=1
                no_question = len(game.questions)
                if (no_question>t_max_failure):
                    t_max_failure = no_question
        couples_t_question_success = [(i,0,0) for i in range (0,t_max_success)]
        couples_t_question_failure = [(i,0,0) for i in range (0,t_max_failure)]
        print("nombre de games success" + str(game_suc))
        print("nombre de games failure" + str(game_fail))
        for game in games:
            if (game.status=="success"):
                no_question = len(game.questions)
                for t in range(no_question):
                    len_q_t = len(game.questions[t])
                    old_average = couples_t_question_success[t][1]
                    old_n = couples_t_question_success[t][2]
                    couples_t_question_success[t] = (t, (old_average*old_n+len_q_t)/(old_n+1),old_n+1)
            if (game.status=="failure"):
                no_question = len(game.questions)
                for t in range(no_question):
                    len_q_t = len(game.questions[t])
                    old_average = couples_t_question_failure[t][1]
                    old_n = couples_t_question_failure[t][2]
                    couples_t_question_failure[t] = (t, (old_average*old_n+len_q_t)/(old_n+1),old_n+1)
                
        
        couples_t_question_success_new = [(couples_t_question_success[t][0],couples_t_question_success[t][1]) for t in range(0,t_max_success)]
        couples_t_question_failure_new = [(couples_t_question_failure[t][0],couples_t_question_failure[t][1]) for t in range(0,t_max_failure)]
        
        couples_t_question_success_new = np.array(couples_t_question_success_new)
        couples_t_question_failure_new = np.array(couples_t_question_failure_new)
        
        
        
        sns.set(style="white")
        sns.regplot(x=couples_t_question_success_new[:, 0], y=couples_t_question_success_new[:, 1], x_ci=None,   label="Succes", marker="o", line_kws={'linestyle':'-'})
        f=sns.regplot(x=couples_t_question_failure_new[:, 0], y=couples_t_question_failure_new[:, 1], x_ci=None, label="Failure", marker="o", line_kws={'linestyle':'-'})

        
        f.legend(loc="best", fontsize='x-large')
        f.set_xlim(0,max(t_max_success, t_max_failure))
        #f.set_ylim(0,20)
        f.set_xlabel("Lenght of questions", {'size':'14'})
        f.set_ylabel("Time", {'size':'14'})
        
        
        """
        plt.plot(couples_t_question_success_new[:, 0], couples_t_question_success_new[:, 1],   label="Succes")
        plt.plot(couples_t_question_failure_new[:, 0], couples_t_question_failure_new[:, 1],   label="Failure")
        plt.xlim(0,max(t_max_success, t_max_failure))
        plt.ylabel("Lenght of questions", {'size':'14'})
        plt.xlabel("Time", {'size':'14'})
        plt.legend()
        """
        #sns.regplot(x=couples_t_question_success_new[:, 0], y=couples_t_question_success_new[:, 1], x_ci=None, x_bins=20, order=4,  label="Succes", marker="o", line_kws={'linestyle':'-'})





