

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


class PersonWord(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(PersonWord, self).__init__(path, self.__class__.__name__, suffix)
        person_categories = ["person"]
        all_categories=[]
        person_used_in = 0
        person_used_out = 0
        person_unused_in = 0
        person_unused_out = 0
        
        for game in games:
            q = game.questions[0]
            words = re.findall(r'\w+', q)
            person_used=False
            for w in words:
                if (w.lower()=="person"):
                    person_used=True
            person_in = False 
            for obj in game.objects : 
                #if( obj.category not in all_categories):
                    #all_categories.append(obj.category)
                if (obj.category in person_categories):
                    person_in=True
            if (person_used == True and person_in==True):
                person_used_in+=1
            if (person_used == True and person_in==False):
                person_used_out+=1
            if (person_used == False and person_in==True):
                person_unused_in+=1
            if (person_used == False and person_in==False):
                person_unused_out+=1
        #print(all_categories)
        #print(len(all_categories))
        labels = 'Person in first question and image','Person in first question, not in image','Person not in first question and in image','Person not in first question and not in image'
        colors = ['yellowgreen', 'gold', 'lightskyblue','lightcoral']
        plt.title("Why the first question is always 'Is it a person?'")
        plt.pie([ person_used_in,person_used_out,person_unused_in,person_unused_out], labels=labels, colors=colors,  autopct='%1.1f%%',shadow=True, startangle=90)

        plt.axis('equal')     

                