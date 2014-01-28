from math import exp, tanh
from random import random
import numpy as np
import time

# funcao nao-linear e sua derivada
def sigmoid(z):
    return 1./(1. + np.exp(-z))
def dsigmoid(z):
    return np.dot(z,(1-z))

# funcao linear e sua derivada
def linear(z):
    return z
def dlinear(z):
    return np.ones(z.shape)

# funcao utilizada na camada escondida e sua derivada
def fhidden(z):
    return sigmoid(z)
def dfhidden(z):
    return dsigmoid(z)

# funcao utilizada na camada de saida e sua derivada
def fout(z):
    return linear(z)
def dfout(z):
    return dlinear(z)

# funcao que produz as saidas da mlp na camada escondida (f)
# e camada de saida (fo), dado uma entrada x e
# os pesos wi (da entrada para escondida) e wh (da escondida para saida)
def runmlp(x,wi,wh):

    ni = len(x)
    nh = len(wi)
    no = len(wh)

    f = np.zeros(nh)
    z = np.dot(wi,x)
    f = fhidden(z)

    fo = np.zeros(no)
    z = np.dot(wh,f)
    fo = fout(z)
        
    return f, fo

# dado as saidas de cada camada (f,fo), a saida desejada (y) e os pesos
# faz a retropropagacao dos erros
def backpropagate(y,f,fo,wi,wh):

    nh = len(wi)
    no = len(wh)

    # o delta da saida eh: (fo-y)*derivada de fo
    deltao = np.dot(fo-y,dfout(fo))

    # o delta da escondida eh: soma de wh[i][.]*deltao[i] multiplicado pela derivada de f
    deltah = np.zeros(nh)
    
    for j in range(nh):
        sdeltao = np.sum( np.dot(deltao,wh[:,j]) )
        deltah[j] = sdeltao*dfhidden(f[j])
    

    return deltah, deltao

# tendo os deltas, basta agora aplicar a funcao de aprendizagem
# em cada peso: w = w - alpha*delta*f
def learn( wi, wh, alpha, deltah, deltao, f, fo, x ):
    
    wh = wh - alpha*np.outer(deltao,f)
    wi = wi - alpha*np.outer(deltah,x)

    return wi, wh

def mlp(ni,nh,no,alpha, X,Y):

    wi = np.random.rand(nh,ni+1)
    wh = np.random.rand(no,nh)
    start = time.time()
    for it in range(1000):
        print it / 1000.0 * 100, "%", time.time() - start
        for x,y in zip(X,Y):
            x = np.array([1]+x)
            f, fo = runmlp(x,wi,wh)
            deltah, deltao = backpropagate(y,f,fo,wi,wh)
            wi, wh = learn( wi, wh, alpha, deltah, deltao, f, fo, x)
                       

    return wi,wh

#X = [[0,0],[0,1],[1,0],[1,1]]
#Y = [[0],[1],[1],[0]]
#wi,wh = mlp(2,3,1,0.5,X,Y)
#print wh.shape
#print wi,wh

#for x,y in zip(X,Y):
#    f, fo = runmlp([1]+x,wi,wh)
#    print x, y, fo
