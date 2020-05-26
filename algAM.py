# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:00:23 2018

@author: Eryka Freires
"""

import numpy as np
import random
from datetime import datetime

# Importando as classes dos modelos
from KNN import KNN
from DMC import DMC
from NaiveBayes import NaiveBayes

bases = {0:"coluna_vertebral_2C.txt",1:"coluna_vertebral_3C.txt",2:"iris.txt",3:"dermatologia.txt",4:"breast_cancer.txt"}

# Carregar base de dados
def carregarBase(escolha):
    dados = np.loadtxt("bases/"+bases[escolha], delimiter = ",")
    return dados

# Normalizar os dados
def normalizar(dados):
    base = np.array(dados)
    minimo = base[:,:-1].min()
    maximo = base[:,:-1].max()
    nova = (base - minimo) / (maximo - minimo)
    nova[:,-1] = base[:,-1]
    return (nova)

# Separar base em treinamento (80%) e teste (20%)
def trein_test(base):
    # Mistura a base de dados
    index = random.sample(list(range(len(base))), len(base))
    q_trein = int(len(base)*0.8)
    nova_base = np.zeros(base.shape)
    for i in range(0,len(base)):
        nova_base[i] = base[index[i]]    
    # Separa em base de treinamento e base de teste
    trein = nova_base[0:q_trein,:]
    test = nova_base[q_trein:len(base),:]
    return (trein,test)

def classes(dados):
    classes = []
    for i in range(0, len(dados)):
        c = dados[i][-1]
        if not(c in classes):
            classes.append(c)
    classes = np.array(classes, dtype=int)
    classes = np.sort(classes)
    print("Classes:", classes)
    return classes

def elem_classes(dados,cla):
    count = 0
    for i in range(0, len(dados)):
        base = dados[i][-1]
        if base == cla:
            count += 1
    return count

def executar(iteracoes,b):
    print("Executando os algoritmos ...")
    # Armazenar matrizes de confusão e taxas de acertos
    knn_taxas = []
    knn_mat_conf = []
    dmc_taxas = []
    dmc_mat_conf = []
    nb_taxas = []
    nb_mat_conf = []
    k_viz = [10,16,17,15,10]
     
    # Carregar base de dados
    base =carregarBase(b)
    # Normalizar base de dados
    base_norm = normalizar(base)
    # Quantidade de classes
    clas = classes(base_norm)
    
    for i in range(0,iteracoes):
        # Separar base em treinamento e teste
        (trein, test) = trein_test(base_norm)
        
        # Treinamento e teste do kNN
        k=k_viz[b]
        knn = KNN(k,trein,test,clas)
        knn.treinamento()
        knn.teste()
        knn_mat_conf.append(knn.matrizConfusao())
        knn_taxas.append(knn.taxaAcertos())
        
        # Treinamento e teste do DMC
        dmc = DMC(trein,test,clas)
        dmc.treinamento()
        dmc.teste()
        dmc_mat_conf.append(dmc.matrizConfusao())
        dmc_taxas.append(dmc.taxaAcertos())
        
        # Treinamento e teste do Naive Bayes
        nb = NaiveBayes(trein,test,clas)
        nb.treinamento()
        nb.teste()
        nb_mat_conf.append(nb.matrizConfusao())
        nb_taxas.append(nb.taxaAcertos())
    
    # Calculos de Médias e Desvio Padrão dos algoritmos
    knn_taxas = np.array(knn_taxas)
    knn_mat_conf = np.array(knn_mat_conf)
    print ("\nResultados KNN: ", k, " \nAcurácia:", knn_taxas.mean()," | Desvio Padrão:", knn_taxas.std()," | Maior Acerto: ",knn_taxas.max())
    print ("Matriz de Confusão da Maior Taxa:\n", knn_mat_conf[knn_taxas.argmax()],"\n")
        
    dmc_taxas = np.array(dmc_taxas)
    dmc_mat_conf = np.array(dmc_mat_conf)
    print ("\nResultados DMC: \nAcurácia:", dmc_taxas.mean()," | Desvio Padrão:", dmc_taxas.std()," | Maior Acerto: ",dmc_taxas.max())
    print ("Matriz de Confusão da Maior Taxa:\n", dmc_mat_conf[dmc_taxas.argmax()],"\n")
        
    nb_taxas = np.array(nb_taxas)
    nb_mat_conf = np.array(nb_mat_conf)
    print ("\nResultados Naive Bayes: \nAcurácia:", nb_taxas.mean()," | Desvio Padrão:", nb_taxas.std()," | Maior Acerto: ",nb_taxas.max())
    print ("Matriz de Confusão da Maior Taxa:\n", nb_mat_conf[nb_taxas.argmax()],"\n")
    
"""
Saída dos algoritmos
Autora: Eryka Freires
Disciplina: Aprendizagem de Máquina
Professor: Ajalmar Rego
"""

print ("Por: Eryka Freires\n", datetime.now(),"\n")
ite = 50 
print ("Quantidade de iterações: ", ite)
for i in range(5):
    print ("----> Base da Dados: ",bases[i],"\n")
    executar(ite,i)