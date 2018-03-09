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

class SuccessNoWordRepetitions(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        
        super(SuccessNoWordRepetitions, self).__init__(path, self.__class__.__name__, suffix)
        
        
        Seuil=20

        status_count = collections.Counter()
        status_list = []

        repetitions = []
        MaxRep=0
        
        stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

        for game in games:

            
            qs=game.questions
            q_counter=collections.Counter()
            tot_words=0
            for q in qs:
                cur = re.sub('[?]', '', q)     
                words = re.findall(r'\w+', cur)
                for w in words:
                    if w.lower() not in stopwords:
                        q_counter[w.lower()]+=1
                        tot_words+=1
                
            Nrep=0
            for w in q_counter:
                Nrep+= q_counter[w]-1
            #srep=0.0
            #if(tot_words!=0):
             #   srep=1.0*Nrep/tot_words
            if(Nrep<Seuil):
                repetitions.append(min(Nrep,Seuil))
                status_count[game.status] += 1
                status_list.append(game.status)
            MaxRep=max(MaxRep,Nrep)
            MaxRep=min(MaxRep,Seuil)
            #repetitions.append(srep)
            #MaxRep=max(MaxRep,srep)
        #print(repetitions)
        
        print("MaxRep vaut"+ str(MaxRep))
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
        f.set_xlabel("words' repetitions", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})
        f.legend(loc="best", fontsize='large')



        ###########################


        success = np.array([s == "success" for s in status_list])
        failure = np.array([s == "failure" for s in status_list])
        incomp  = np.array([s == "incomplete" for s in status_list])


        repetitions = np.array(repetitions)
        #rng = np.arange(0.0, MaxRep,MaxRep/20)
        rng=range(0,MaxRep)
        #print(MaxRep)
        histo_success = np.histogram(repetitions[success], bins=rng)
        histo_failure = np.histogram(repetitions[failure], bins=rng)
        histo_incomp  = np.histogram(repetitions[incomp], bins=rng)


        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        
        #print(histo_success[0])
        #print(normalizer)
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer
        print("Max rep :" +str(MaxRep))
        print("Normalizer :" +str(normalizer) +" with lenght "+str(len(normalizer)))


        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        #print(curve)
        f = sns.regplot(x=np.linspace(1, 20, len(curve)), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})



