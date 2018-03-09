

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

stopwords = ["a", "an", "is", "it", "the", "does", "do", "are", "you", "that",
             "they", "doe", "this", "there", "hi", "his", "her", "its", "picture", "can", "he", "she", "bu", "us", "photo"]

class WordType(AbstractPlotter):
    def __init__(self, path, games, logger, suffix):
        super(WordType, self).__init__(path, self.__class__.__name__, suffix)
        
        
        color_list =["green",'blue', 'brown', "red", 'white', "black", "yellow", "color", "orange", "pink"]
        people_list  =['people', 'person', "he", "she", "human", "man", "woman", "guy", 'alive', "girl", "boy", "head", 'animal']
        prep = ['on', "in", 'of', 'to', "with", "by", "at", "or", "and", "from"]
        number = ['one', "two", "three", "four", "five", "six", "first", "second", "third", "half"]
        spatial = ["top", "left", "right", "side", "next", "front", "middle", "foreground", "bottom", "background",
                   "near", "behind", "back", "at", "row", "far", "whole", "closest"]
        verb=["wearing", "have", "can", "holding", "sitting", "building", "standing", "see"]
        obj = ["hand","table", 'car', "food", "plate", "shirt", "something", "thing", "object",
               "light", "hat", "tree", "bag", "book", "sign", "bottle", "glass", "bus", "wall", "vehicle",
               "chair", "dog", "cat", "windows", "boat", "item", "shelf", "horse", "furniture", "water", "camera", "bike",
               "train", "window", "bowl", "plant", "ball", "cup", ]
        misc = [ 'visible', "made", "part", "piece", "all"]

        
            
        n_colors=0
        n_peoples=0
        n_preps=0
        n_numbers=0
        n_spatials=0
        n_verbs=0
        n_objs=0
        n_miscs=0
        
        for game in games:
            no_questions = len(game.questions)
            for t in range(no_questions):
                q = game.questions[t]
                words = re.findall(r'\w+', q)
                for w in words:
                    w=w.lower()
                    if w in color_list: 
                        n_colors+=1
                    if w in people_list: 
                        n_peoples+=1
                    if w in prep: 
                        n_preps+=1
                    if w in number: 
                        n_numbers+=1
                    if w in spatial: 
                        n_spatials+=1
                    if w in verb: 
                        n_verbs+=1
                    if w in obj: 
                        n_objs+=1
                    if w in misc: 
                        n_miscs+=1
        
        labels = 'color words','people','prep','number','spatial','verb','obj','misc'
        colors = ['yellowgreen', 'gold', 'lightskyblue','lightcoral','mediumslateblue','lightgray','peru','c','m']
        plt.title("Distribution of word use in the dialogues")
        plt.pie([n_colors,n_peoples,n_preps,n_numbers,n_spatials,n_verbs,n_objs,n_miscs], labels=labels, colors=colors,  autopct='%1.1f%%',shadow=True, startangle=90)

        plt.axis('equal')     
                