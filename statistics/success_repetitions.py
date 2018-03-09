import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessRepetitions(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessRepetitions, self).__init__(path, self.__class__.__name__, suffix)

        status = []
        percent_repetition_list = [] #number of objects whose category is repeted/number total of objects
        status_count = collections.defaultdict(int)


        for game in games:

            status_count[game.status] += 1
            status.append(game.status)
            categories = []
            nb=0
            for obj in game.objects : 
                if (obj.category in categories):
                    nb+=1
                else :
                    categories.append(obj.category)
            percent_repetition_list.append(nb/len(game.objects))


        success = np.array([s == "success" for s in status])
        failure = np.array([s == "failure" for s in status])
        incomp  = np.array([s == "incomplete" for s in status])

        #print(area_list[:5])

        sns.set(style="whitegrid", color_codes=True)

        

        if sum(incomp) > 0:
            columns = ['Repetitions', 'Success', 'Failure', 'Incomplete']
            data = np.array([np.array(percent_repetition_list), success, failure, incomp]).transpose()
        else:
            columns = ['Repetitions', 'Success', 'Failure']
            data = np.array([np.array(percent_repetition_list), success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby(pd.cut(df["Repetitions"], np.arange(0,1,0.1))).sum()
        df = df.drop('Repetitions', 1)
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(9,6), color=["g", "r", "b"])

 
        f.set_xlabel("nb of objects of the same category than the one of interest", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})



        f.legend(loc="upper left", fontsize='x-large')
        
  
###########################################
        percent_repetition_list = np.array(percent_repetition_list)
        rng = np.arange(0,1,0.1)
        histo_success = np.histogram(percent_repetition_list[success], bins=rng)
        histo_failure = np.histogram(percent_repetition_list[failure], bins=rng)
        histo_incomp  = np.histogram(percent_repetition_list[incomp] , bins=rng)

        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer

        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        
        f = sns.regplot(x=np.linspace(0,9,9), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})

