# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:48:51 2018

@author: Eryka Freires
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import math

class DMC:
    # Construtor da classe DMC
    def __init__(self,trein, test,clas):
        self.trein = trein
        self.test = test
        self.clas = clas

    def pos_class(self,c):
        for a in range(0,len(self.clas)):
            if self.clas[a] == c:
                return a  
        return 0

    # Calcula a distância Euclidiana
    def dist_euclidiana(self,d1,d2):
        dist = math.sqrt(sum((d1-d2)**2))
        return dist

    # Calcula a distância Manhattan
    def dist_manhattan(self,d1,d2):
        dist = sum(np.abs((d1-d2)))
        return dist

    
    def treinamento(self): 
        atrib = len(self.trein[0][:-1])
        self.med_atr = np.zeros((len(self.clas),atrib))
        q_atri = np.zeros(len(self.clas))
        for a in range(0,len(self.trein)):
            c = int(self.trein[a][-1:])
            pos = self.pos_class(c)
            for v in range(0,atrib):
                self.med_atr[pos][v] += self.trein[a][v]
            q_atri[pos] += 1
        for c in range(0,len(self.clas)):
                self.med_atr[c][:] /= q_atri[c]

    def teste(self):
        classe = 0
        self.rot_verd = np.zeros(len(self.test))
        self.rot_prev = np.zeros(len(self.test))
        for a in range(0,len(self.test)):
            dist = np.zeros(len(self.clas))
            dado = self.test[a][:-1]
            for c in range(0,len(self.clas)):
                d2 = self.med_atr[c][:]
                dist[c] = self.dist_euclidiana(dado,d2)
            classe = self.clas[np.argmin(dist)]
            self.rot_verd[a] = self.test[a][-1:]
            self.rot_prev[a] = classe
    
    def matrizConfusao(self):
        mat_conf = []
        mat_conf.append(confusion_matrix(self.rot_verd, self.rot_prev))
        mat_conf = np.array(mat_conf)
        return mat_conf
    
    def taxaAcertos(self):
        acertos = 0
        taxa = 0
        for a in range(0,len(self.test)):
            if (self.rot_verd[a] == self.rot_prev[a]):
                acertos = acertos + 1
        taxa = acertos/len(self.test)
        return taxa