#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:01:29 2018

@author: c.senik
"""



import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd
import re

class SuccessNoRepetitions(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessNoRepetitions, self).__init__(path, self.__class__.__name__, suffix)

        status_count = collections.Counter()
        status_list = []

        repetitions = []
        MaxRep=0

        for game in games:

            status_count[game.status] += 1
            status_list.append(game.status)
            qs=game.questions
            q_counter=collections.Counter()
            for q in qs:
                cur = re.sub('[?]', '', q)                
                q_counter[cur]+=1
            Nrep=0
            for w in q_counter:
                Nrep+= q_counter[w]-1
            

            repetitions.append(Nrep)
            MaxRep=max(MaxRep,Nrep)

        sns.set(style="whitegrid", color_codes=True)

        success = np.array([s == "success" for s in status_list]) + 0
        failure = np.array([s == "failure" for s in status_list]) + 0
        incomp  = np.array([s == "incomplete" for s in status_list]) + 0



        if sum(incomp) > 0:
            columns = ['No repetitions', 'Success', 'Failure', 'Incomplete']
            data = np.array([repetitions, success, failure, incomp]).transpose()
        else:
            columns = ['No repetitions', 'Success', 'Failure']
            data = np.array([repetitions, success, failure]).transpose()


        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('No repetitions').sum()
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, color=["g", "r", "b"])

        sns.regplot(x=np.array([0]), y=np.array([0]), scatter=False, line_kws={'linestyle':'--'}, label="% Success",ci=None, color="b")


        #f.set_xlim(0.5,18.5)
        #f.set_ylim(0,25000)
        f.set_xlabel("Number of repetitions", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})
        f.legend(loc="best", fontsize='large')



        ###########################


        success = np.array([s == "success" for s in status_list])
        failure = np.array([s == "failure" for s in status_list])
        incomp  = np.array([s == "incomplete" for s in status_list])


        repetitions = np.array(repetitions)
        rng = range(0, MaxRep)
        histo_success = np.histogram(repetitions[success], bins=rng)
        histo_failure = np.histogram(repetitions[failure], bins=rng)
        histo_incomp  = np.histogram(repetitions[incomp], bins=rng)


        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer


        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        f = sns.regplot(x=np.linspace(0, MaxRep, len(curve)), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})



