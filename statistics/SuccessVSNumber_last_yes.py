#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:38:28 2018

@author: c.senik
"""


import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

def last_yes(g):
    answers=g.answers
    num_yes=0
    n= len(answers)
    while (n>0 and answers[n-1].lower()=="yes"):
        num_yes+=1
        n-=1
    #print(num_yes)
    return num_yes

class SuccessNumberLastYes(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessNumberLastYes, self).__init__(path, self.__class__.__name__, suffix)

        status_list = []
        status_count = collections.defaultdict(int)
        number_yes = []

        for game in games:

            number_yes.append(last_yes(game))

            status_count[game.status] += 1
            status_list.append(game.status)


        success = np.array([s == "success" for s in status_list]) + 0
        failure = np.array([s == "failure" for s in status_list]) + 0
        incomp  = np.array([s == "incomplete" for s in status_list]) + 0

        sns.set_style("whitegrid", {"axes.grid": False})

        if sum(incomp) > 0:
            columns = ['Number_Yes', 'Success', 'Failure', 'Incomplete']
            data = np.array([number_yes, success, failure, incomp]).transpose()
        else:
            columns = ['Number_Yes', 'Success', 'Failure']
            data = np.array([number_yes, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Number_Yes').sum()
        df = df.div(df.sum(axis=1), axis=0)
        #df = df.sort_values(by='Success')
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3)

        f.set_xlim(-0.5,29.5)

        plt.xlabel("Number of yes ending the dialogue", {'size':'14'})
        plt.ylabel("Success ratio", {'size':'14'})
