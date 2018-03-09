import collections
import numpy as np
from guesswhat.statistics.abstract_plotter import *

import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt

class SuccessDmin(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(SuccessDmin, self).__init__(path, self.__class__.__name__, suffix)

        status = []
        dmin_list = []
        status_count = collections.defaultdict(int)


        for game in games:

            status_count[game.status] += 1
            status.append(game.status)
            game_objects = game.objects
            dmin2=2
            for obj in game_objects : 
                obj_bbox = obj.bbox
                picture = game.image
                x_1 = obj_bbox.x_center/ picture.width  #eventuellement picture.weught pour tout ramener entre 0 et 1 
                y_1 = obj_bbox.y_center / picture.height
                for obj2 in game_objects : 
                    if(obj2.id !=obj.id):
                        obj2_bbox = obj2.bbox
                        x2 = obj2_bbox.x_center/ picture.width  #eventuellement picture.weught pour tout ramener entre 0 et 1 
                        y2 = obj2_bbox.y_center / picture.height
                        if ((x2-x_1)**2+(y2-y_1)**2 < dmin2):
                            dmin2=(x2-x_1)**2+(y2-y_1)**2 
            dmin_list.append(dmin2)


        success = np.array([s == "success" for s in status])
        failure = np.array([s == "failure" for s in status])
        incomp  = np.array([s == "incomplete" for s in status])


        sns.set(style="whitegrid", color_codes=True)

        

        if sum(incomp) > 0:
            columns = ['Area', 'Success', 'Failure', 'Incomplete']
            data = np.array([dmin_list, success, failure, incomp]).transpose()
        else:
            columns = ['Area', 'Success', 'Failure']
            data = np.array([dmin_list, success, failure]).transpose()

        df = pd.DataFrame(data, columns=columns)
        df = df.convert_objects(convert_numeric=True)
        df = df.groupby(pd.cut(df["Area"], np.arange(0,0.02,0.002))).sum()
        df = df.drop('Area', 1)
        f = df.plot(kind="bar", stacked=True, width=1, alpha=0.3, figsize=(9,6), color=["g", "r", "b"])

 
        f.set_xlabel("Minimal distance between objects of the image", {'size':'14'})
        f.set_ylabel("Number of dialogues", {'size':'14'})



        f.legend(loc="upper left", fontsize='x-large')
        
        

        ###########################################
        dmin_list = np.array(dmin_list)
        rng = np.arange(0,0.02,0.002)
        histo_success = np.histogram(dmin_list[success], bins=rng)
        histo_failure = np.histogram(dmin_list[failure], bins=rng)
        histo_incomp  = np.histogram(dmin_list[incomp] , bins=rng)

        normalizer = histo_success[0] + histo_failure[0] + histo_incomp[0]
        histo_success = 1.0*histo_success[0] / normalizer
        histo_failure = 1.0*histo_failure[0] / normalizer
        histo_incomp  = 1.0*histo_incomp[0]  / normalizer

        ax2 = f.twinx()
        curve = np.ones(len(normalizer))-histo_failure-histo_incomp
     
        f = sns.regplot(x=np.linspace(1, 10, 9), y=curve, order=3, scatter=False, line_kws={'linestyle':'--'},ci=None, truncate=False, color="b")
        ax2.set_ylim(0,1)
        ax2.grid(None)
        ax2.set_ylabel("Success ratio", {'size':'14'})
        


