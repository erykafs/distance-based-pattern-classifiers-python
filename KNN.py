# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:48:35 2018

@author: Eryka Freires
"""
from sklearn.metrics import confusion_matrix
import numpy as np
import math

class KNN:
    # Construtor da classe KNN
    def __init__(self,k, trein, test, clas):
        self.k = k
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
        d1 = np.array(d1)
        d2 = np.array(d2)
        dist = math.sqrt(sum((d1-d2)**2))
        return dist

    # Calcula a distância Manhattan
    def dist_manhattan(self,d1,d2):
        dist = sum(np.abs((d1-d2)))
        return dist

    def treinamento(self):
        return 0
    
    def teste(self):
        self.rot_verd = np.zeros(len(self.test))
        self.rot_prev = np.zeros(len(self.test))
        for a in range(0,len(self.test)):
            dists, tam_trein = {}, len(self.trein)
            p2 = self.test[a][:-1]
            for i in range(0,len(self.trein)):
                p1 = self.trein[i][:-1]
                dist_aux = self.dist_euclidiana(p1,p2)
                dists[i] = dist_aux    
            # Obter as chaves dos k-vizinhos mais próximos
            k_viz = sorted(dists, key=dists.get)[:self.k]              
            vot_maj = np.zeros(len(self.clas), dtype=int)
            for v in k_viz:
                #pos = int(self.trein[v][-1:]) - 1
                #vot_maj[pos] = vot_maj[pos] + 1
                c = int(self.trein[v][-1:])
                pos = self.pos_class(c)
                vot_maj[pos] = vot_maj[pos] + 1
            classe = self.clas[np.argmax(vot_maj)]
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