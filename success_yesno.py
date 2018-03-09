
import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessYesNo(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessYesNo, self).__init__(path, self.__class__.__name__, suffix)

        status = []
        yesno_list = [] #store for each game the ratio yes/(yes+no)
        status_count = collections.defaultdict(int)


        for game in games:

            status_count[game.status] += 1
            status.append(game.status)
            yes=0
            no=0
            for a in game.answers:

                a = a.lower()
                if a == "yes":
                    yes +=1
                elif a == "no":
                    no += 1
            
            if (yes+no == 0 ):
                yesno_list.append(int(0))
            else:
                yesno_list.append(int(100*yes/(yes+no)))
        """     
        success = np.array([s == "success" for s in status]) + 0
        failure = np.array([s == "failure" for s in status]) + 0
        incomp  = np.array([s == "incomplete" for s in status]) + 0

        sns.set_style("whitegrid", {"axes.grid": False})

        if sum(incomp) > 0:
            columns = ['Percent of Yes', 'Success', 'Failure', 'Incomplete']
            data = np.array([yesno_list, success, failure, incomp]).transpose()
        else:
            columns = ['Percent of Yes', 'Success', 'Failure']
            data = np.array([yesno_list, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby('Percent of Yes').sum()
        df = df.div(df.sum(axis=1), axis=0)
        #df = df.sort_values(by='Success')
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3)

        f.set_xlim(0,100)

        plt.xlabel("Percent of Yes", {'size':'14'})
        plt.ylabel("Success ratio", {'size':'14'})
        """

       
        success = np.array([s == "success" for s in status])
        failure = np.array([s == "failure" for s in status])
        incomp  = np.array([s == "incomplete" for s in status])



        sns.set(style="whitegrid", color_codes=True)

        

        if sum(incomp) > 0:
            columns = ['YesNo', 'Success', 'Failure', 'Incomplete']
            data = np.array([yesno_list, success, failure, incomp]).transpose()
        else:
            columns = ['YesNo', 'Success', 'Failure']
            data = np.array([yesno_list, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby(pd.cut(df["YesNo"], 10)).sum()
        df = df.drop('YesNo', 1)
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(9,6), color=["g", "r", "b"])

     
        f.set_xlabel("% of yes", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})
        

        #sns.regplot(x=np.array([0]), y=np.array([0]), scatter=False, line_kws={'linestyle':'--'}, label="% Success",ci=None, color="b")

        f.legend(loc="upper left", fontsize='x-large')
        
        #dont work for the final because some intervals on abcisses have 0 coordinates

        ###########################################
        rng = np.arange(0,100)
        histo_success = np.histogram(np.asarray(yesno_list)[success], bins=rng)
        histo_failure = np.histogram(np.asarray(yesno_list)[failure], bins=rng)
        histo_incomp  = np.histogram(np.asarray(yesno_list)[incomp] , bins=rng)

        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer


        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        try : 
            f = sns.regplot(x=np.linspace(1, 10, 8), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        except ValueError:  #raised if `y` is empty.
            pass
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})
        

