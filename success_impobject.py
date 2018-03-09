
import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd

class SuccessImpObject(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessImpObject, self).__init__(path, self.__class__.__name__, suffix)

        status = []
        area_list = []
        status_count = collections.defaultdict(int)


        for game in games:

            status_count[game.status] += 1
            status.append(game.status)
            area_max=0
            for obj in game.objects:
                if (obj.area>area_max):
                    area_max = obj.area

            area_list.append(float(game.object.area)*100/(area_max))


        success = np.array([s == "success" for s in status])
        failure = np.array([s == "failure" for s in status])
        incomp  = np.array([s == "incomplete" for s in status])

        #print(area_list[:5])

        sns.set(style="whitegrid", color_codes=True)

        rng = [0,5,10,20,30,50,80]

        if sum(incomp) > 0:
            columns = ['Area', 'Success', 'Failure', 'Incomplete']
            data = np.array([np.array(area_list), success, failure, incomp]).transpose()
        else:
            columns = ['Area', 'Success', 'Failure']
            data = np.array([np.array(area_list), success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby(pd.cut(df["Area"], [0,5,10,20,30,50])).sum()
        df = df.drop('Area', 1)
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(9,6), color=["g", "r", "b"])

        #f.set_xlim(-0.5,8.5)
        #f.set_ylim(0,30000)
        f.set_xlabel("log of object area", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})



        f.legend(loc="upper left", fontsize='x-large')
        
        

        ###########################################
        area_list = np.array(area_list)
        histo_success = np.histogram(area_list[success], bins=rng)
        histo_failure = np.histogram(area_list[failure], bins=rng)
        histo_incomp  = np.histogram(area_list[incomp] , bins=rng)

        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer

        ax2 = f.twinx()

        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
        
        f = sns.regplot(x=np.linspace(1, 10, 6), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})


