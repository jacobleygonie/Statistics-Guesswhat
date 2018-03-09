

import json
from pprint import pprint
import itertools
import collections

import wordcloud as wc

import re
import sys
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from guesswhat.statistics.abstract_plotter import *


class ThirdQuestionCategory(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(ThirdQuestionCategory, self).__init__(path, self.__class__.__name__, suffix)
        #this list is the list of all categories for the train 
        all_categories=['person', 'backpack', 'frisbee', 'chair', 'hot dog', 'cup', 'sink', 'dining table', 'vase', 'handbag', 'bowl', 'couch', 'refrigerator', 'bottle', 'oven', 'banana', 'orange', 'surfboard', 'microwave', 'spoon', 'tennis racket', 'cake', 'knife', 
                        'carrot', 'sandwich', 'baseball glove', 'baseball bat', 'elephant', 'umbrella', 'car', 'bed', 'motorcycle', 'truck', 'bus', 'cat', 'remote', 'potted plant', 'traffic light', 'bench', 'fork', 'parking meter', 'wine glass', 'laptop', 'toilet', 
                        'dog', 'tie', 'suitcase', 'donut', 'cow', 'boat', 'horse', 'teddy bear', 'sports ball', 'snowboard', 'pizza', 'apple', 'book', 'sheep', 'bicycle', 'clock', 'toothbrush', 'train', 'hair drier', 'scissors', 'tv', 'bird', 'broccoli', 'kite', 'cell phone',
                        'keyboard', 'skateboard', 'airplane', 'mouse', 'skis', 'fire hydrant', 'zebra', 'giraffe', 'bear', 'stop sign', 'toaster']
        
        category_used_in_success = 0
        category_used_out_success = 0
        no_category_success = 0
        category_used_in_failure = 0
        category_used_out_failure = 0
        no_category_failure = 0
        
        for game in games:
            for obj in game.objects : 
                if(obj.category not in all_categories):
                    all_categories.append(obj.category)
                    
        for game in games:
            if (len(game.questions)>2):
                q = game.questions[2]
                words = re.findall(r'\w+', q)
                category_used=False
                possible_category = ""
                for w in words:
                    if (w.lower() in all_categories):
                        category_used=True
                        possible_category = w.lower()
                category_in = False 
                if(category_used):
                    for obj in game.objects :
                        if (obj.category ==possible_category):
                            category_in=True
          
                if (category_used == True and category_in==True and game.status=="success"):
                    category_used_in_success+=1
                if (category_used == True and category_in==True and game.status=="failure"):
                    category_used_in_failure+=1
                if (category_used == True and category_in==False and game.status=="success"):
                    category_used_out_success+=1  
                if (category_used == True and category_in==False and game.status=="failure"):
                    category_used_out_failure+=1    
                if (category_used ==False and  game.status=="success"):
                    no_category_success+=1
                if (category_used ==False and  game.status=="failure"):
                    no_category_failure+=1
                
        
        #print(all_categories)
        #print(len(all_categories))
        labels = 'used, in, success','used, in, failure','used, not in, success','used, not in, failure', 'not used, success', 'not_used, failure'
        colors = ['yellowgreen', 'gold', 'lightskyblue','lightcoral','lightgray','peru']
        plt.title("Is a category used as a word in the third question ? Is this category is in the image ? Success ? ")
        plt.pie([ category_used_in_success,category_used_in_failure,category_used_out_success,category_used_out_failure,no_category_success,no_category_failure], labels=labels, colors=colors,  autopct='%1.1f%%',shadow=True, startangle=90)

        plt.axis('equal')     

                