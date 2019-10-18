import numpy as np 
from kaldi.util import levenshtein_edit_distance
class Scorer(object):
    def __init__(self,wordfile=None):
        self.WordDict = {}
        self.WordList = []
        with open(wordfile,'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split(' ')
                self.WordDict[l[0]] = int(l[1])
                self.WordList.append(l[0])

    def text2lattice(self,text):
        if type(text) == str:
            txt_list = text.split(' ')
        elif type(text) == list:
            txt_list = text
        else:
            txt_list = list()
        lattice = []
        for t in txt_list:
            if t in self.WordDict:
                lattice.append(self.WordDict[t])
            else:
                lattice.append(self.WordDict["<UNK>"])
        return lattice

    def edit_distance(self,x,y):
        if(type(x[0])==str):
            x = self.text2lattice(x)
        if(type(y[0]==str)):
            y = self.text2lattice(y)
        dist = levenshtein_edit_distance(x,y)
        return dist