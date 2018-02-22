'''
Project:
Author: Yilun Zhang
Date: 1/21/18
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from scipy.stats import multivariate_normal

num_iter=100
n_subclass=1
#dimension of problem =2

class_name='RedSofa'

# def calPrior(data):
#
#     return pc, px


def cal_probG(x, A, mu):
    d = len(x[0])
    exp_num=-1./2*np.einsum('ij,jk,ki->i',x-mu,np.linalg.inv(A),(x-mu).T)
    return np.exp(exp_num)/pow((2*np.pi)**d*np.linalg.det(A),0.5)
    # return
def loadTrain(mode):
    mkFl = 'labeled_data'
    mkList = [file for file in os.listdir(mkFl+'/'+class_name) if file.endswith('.npy')]
    if mode == 1:
        imFl = 'images'

    imList= [file for file in os.listdir(imFl+'/'+class_name) if file.endswith('.png')]

    for i in range(len(mkList)):
        im = cv2.imread(os.path.join(imFl,class_name,imList[i]))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2HLS)
        mk = np.load(os.path.join(mkFl,class_name, mkList[i]))
        # im_mked = np.multiply(im,mk)
        t_coor = np.where(mk == True)
        if i ==0:
            data = np.zeros([t_coor[0].shape[0], 2])
            data[:, :] =im[t_coor][:,[0,2]]
        elif i >0:
            data= np.append(data, im[t_coor][:,[0,2]],axis=0)
    return data.astype(np.float64)


def EMAlgorithm(data, n_subclass):
    n_data, d = len(data), len(data[0])
    cri = new_cri = np.finfo(float).eps
    lg=[]
    #initialize alpha, mu, A
    alpha=[]
    mu = []
    A =[]
    for i in range(n_subclass):
        alpha.append(1/n_subclass)
        mu.append(data[n_data//n_subclass*i])
        A.append(np.array([[50., 0],[0, 50.]]))
        # print(np.linalg.det(A))
    r = np.zeros([n_data, n_subclass])

    #finish ini

    for i in range(num_iter):
        # for m in range(n_data):
        #     print(m)
        #     sumAlpG_AllSubCls = 0
        for k in range(n_subclass):
            # print(A[k])
            # r[m, k] = cal_probG(data[m], A[k], mu[k], alpha[k])
            r[:, k] = alpha[k] * cal_probG(data, mu[k], A[k])
        sumAlpG_AllSubCls=np.sum(r,axis=1)

        # if(sumAlpG_AllSubCls==0):
        #     print(1)
        #     r[m, :]=1/3.,1/3.,1/3.
        # else:
        #     r[m,:]/= sumAlpG_AllSubCls ##E-step
        r /= np.array([sumAlpG_AllSubCls, sumAlpG_AllSubCls, sumAlpG_AllSubCls]).T
        r[np.where(sumAlpG_AllSubCls==0.)]=1/3.,1/3.,1/3.
        # print(sumAlpG_AllSubCls)
        # print(r[m, :])
        sum_r=np.sum(r,axis=0)


        new_cri = np.sum(np.log(np.amax(r, axis=1)))#new log likelihood
        print(new_cri)
        print((cri - new_cri) / cri)
        lg.append(new_cri) ## lg var for ploting
        if (abs(cri-new_cri))/abs(cri)<0.0005:
            # print((cri-new_cri)/cri)
            break
        cri = new_cri

        #M-step

        for k in range(n_subclass):
            mu[k]*=0.
            A[k] *= 0.
            mu[k]=np.sum(r[:,k].reshape(-1,1)*data[:],axis=0)
                # print(mu[k])
            mu[k]/=sum_r[k]#new mu
            alpha[k]=1/n_data*sum_r[k]#new alpha
            for m in range(n_data):
                A[k] += (r[m, k] * (data[m]-mu[k]).reshape(2,1)).dot((data[m]-mu[k]).reshape(1,2))
            A[k] /=sum_r[k]   # new A
    return mu, A, alpha, lg





if __name__=='__main__':
    mode=1#1 means train, 0 means test
    n_subclass=3
    data=loadTrain(mode)
    mu, A, alpha, lg = EMAlgorithm(data, n_subclass)
    aa={}

    aa['mu']=mu
    aa['A'] = A
    aa['alpha'] = alpha
    aa['lg'] = lg
    aa['n_subclass'] = n_subclass
    pickle.dump(aa, open(class_name+'.pkl', "wb"))


