#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:03:36 2018

@author: clairelasserre
"""

import json
from pprint import pprint
import copy
data = json.load(open('old_dict.json'))
#pprint(data)
#si on enlève tous les mots spatiaux dans les 30 les plus utilisés
spatial = [ 'person','whole', 'top',  'side', 'one', 'of', 'half', 'right', 'middle', 'on',  'front', 'left', 'in']

dict2 = copy.deepcopy(data)
for w in data["word2i"]:
    if (w in spatial):      
        del dict2["word2i"][w]
        
json = json.dumps(dict2)
f = open("dict.json","w")
f.write(json)
f.close()

