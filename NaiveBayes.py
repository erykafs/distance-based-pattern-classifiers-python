# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:49:57 2018

@author: Eryka Freires
"""

from sklearn.metrics import confusion_matrix
import numpy as np

class NaiveBayes:
    # Construtor da classe NaiveBayes
    def __init__(self,trein, test, clas):
        self.trein = trein
        self.test = test
        self.clas = clas
        
    def elem_classes(self,dados):
        elem_cla = np.zeros(len(self.clas),dtype=int)
        for i in range(0, len(dados)):
            base = int(dados[i][-1])
            for c in range(0,len(self.clas)):
                if base == (c+1):
                    elem_cla[c] += 1 
        return elem_cla

        
    def gaussian(self, x, mu, sig):
        return (np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2))))/(sig * np.sqrt(2*np.pi))

    def gaussianMult(self, x, mu, sig):
        d = len(self.trein)
        mtCovar = np.ones((d,d), dtype=np.float64)
        # Calcular a Matriz de Co-variância
        
        invCovar = np.linalg.inv(mtCovar) # inversa
        detCovar = np.linalg.det(mtCovar) # determinante
        trans = (x - mu).transpose()
        
        return (np.exp(-(trans*invCovar*(x - mu))) / np.sqrt(np.power((2*np.pi), 2)) * np.sqrt(detCovar))   
    
    def treinamento(self):
        # Estrutura para guardar o desvio padrão e média (((qnt_cla, 2, qut_atr)))
        atr_cla = self.elem_classes(self.trein)
        qtn_cla = len(self.clas)
        self.atr = len(self.trein[0][:-1])
        self.metricas = np.zeros(((qtn_cla,2,self.atr)))
        # Gerar toda a matriz, calcular o desvio e a media e jogar na estrutura
        for c in range(0, qtn_cla): # Calcular a média a desvio por classe
            amostras = np.zeros((atr_cla[c],self.atr))
            calc_amos = np.zeros((2,self.atr))
            pos = 0
            for a in range(0,len(self.trein)):
                if (self.trein[a][-1] == (c+1)):
                    amostras[pos][:] = self.trein[a][:-1]
                    pos += 1
            for i in range(0,self.atr):
               calc_amos[0][i] = amostras[:,i].mean()
               calc_amos[1][i] = amostras[:,i].std()
            # Armazena nas métricas
            self.metricas[c][0][:] = calc_amos[0][:]
            self.metricas[c][1][:] = calc_amos[1][:]
        
    def teste(self):
        qt_cla = len(self.clas)
        self.rot_verd = np.zeros(len(self.test))
        self.rot_prev = np.zeros(len(self.test))
        for a in range(0,len(self.test)):
            classe = 0
            prob = np.ones(qt_cla, dtype=np.float64)
            dado = self.test[a][:-1]
            for c in range(0,qt_cla):
                for i in range(0,self.atr):
                    if (self.metricas[c][0][i] != 0) and (self.metricas[c][1][i] != 0):
                        prob[c] *= self.gaussian(dado[i],self.metricas[c][0][i],self.metricas[c][1][i])
            # Calcular a maior probabilidade
            if self.clas[0] == 2:
                print (prob)
            classe = self.clas[np.argmax(prob)]
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