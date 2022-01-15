import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from copy import  copy
from time import time
from collections import defaultdict, Counter
from sklearn.preprocessing import minmax_scale

pd.set_option('display.max_column',500)
pd.set_option('display.float_format','{0:.2f}'.format)

def fun(x):
    return  3.+np.sum((x**2-3.*np.cos(2*np.pi*x)),-1)
a=np.random.rand(10,1)
# print(a.shape)
# print(fun(a))

low,high,size=-5,5,100
spacing=np.linspace(low,high,size)
print("space:",spacing.shape)
xx,yy=np.meshgrid(spacing,spacing)
# print(xx.shape,yy.shape)

grid=np.stack((xx,yy),-1)
# print(grid.shape)
z=fun(grid).T

# from mpl_toolkits.mplot3d import axes3d,Axes3D
# fig=plt.figure(figsize=(7,5))
# ax =Axes3D(fig)
# ax.plot_surface(xx,yy,z,rstride=1,cstride=1,cmap=plt.cm.viridis_r,linewidth=0,antialiased=False)
# plt.show()
#
# extent=np.min(xx)-1,np.max(xx)+1,np.min(yy),np.max(yy)
# print(extent)
# fig,ax=plt.subplots(figsize=(7,7))
# ax.imshow(z,cmap=plt.cm.viridis_r,extent=extent,origin='lowleft')
# plt.show()

function=fun

# n=50
# k=10
# mean=[1.9,-2.7]
# cov=0.5*np.eye(2)
# print(cov)
# x=np.random.multivariate_normal(mean=mean,cov=cov,size=n)
# print(x.shape)
#
# extent=np.min(xx),np.max(xx),np.min(yy),np.max(yy)
# fig,ax=plt.subplots(figsize=(7,7))
# # x=np.random.multivariate_normal(mean=mean,cov=cov,size=n)
# # ax.imshow(z,cmap=plt.cm.viridis_r,extent=extent,origin='lowleft')
# # ax.scatter(x.T[0],x.T[1],color='k',facecolor='none',s=50)
# # ax.scatter(mean[0],mean[1],color='k',s=70)
# # plt.show()
#
# loss=function(x)
# arg_topk=np.argsort(loss)[:k]
# topk=x[arg_topk]
# ax.imshow(z,cmap=plt.cm.viridis_r,extent=extent,origin='lowleft')
# ax.scatter(x.T[0],x.T[1],color='k',facecolor='none',s=50)
# ax.scatter(topk.T[0],topk.T[1],color='k',facecolor='r',s=50)
# ax.scatter(mean[0],mean[1],color='k',s=70)
# plt.show()

# mean_all=topk.T.mean(axis=1)
# print(topk.mean(axis=0))
# print(mean_all)

class CMAES():
    def __init__(self,function,n=50,k=10,mean=np.zeros(2),cov=np.eye(2),alpha=0.5,max_iter=10):
        self.function=function
        self.n=n
        self.k=k
        self.mean_all=mean
        self.cov=cov
        self.old=cov
        self.alpha=alpha
        self.max_iter=max_iter
        self.old_optimum=mean

        self.x=np.random.multivariate_normal(mean=self.mean_all,cov=self.cov,size=self.n)
        self.topk=mean
    def update(self,show=True):
        loss=self.function(self.x)
        arg_topk=list(np.argsort(loss)[:self.k])
        self.topk=self.x[arg_topk]

        if show:
            self.show()
        mean_all=self.x.mean(axis=0)
        center=self.topk-mean_all
        self.old=self.cov
        current=center.T.dot(center)/(self.k-1)
        self.cov=0.5*self.old+0.5*current
        # self.cov=current
        self.mean_topk=self.topk.mean(axis=0)
        self.x=np.random.multivariate_normal(mean=self.mean_topk,cov=self.cov,size=self.n)

    def update1(self, show=True):
        # Calculate the loss of the samples
        losses = self.function(self.x)

        # Keep top-k samples with lowest loss
        arg_topk = list(np.argsort(losses)[:self.k])
        self.topk = self.x[arg_topk]

        if show:
            self.show()

        # Calculate adapted Covariance-matrix
        mean_all = self.x.T.mean(axis=1)
        centered = self.topk - mean_all
        self.cov = centered.T.dot(centered) / (self.k - 1)

        # Take new samples around mean(top−k) samples with a Covariance-matrix C
        self.mean_topk = self.topk.T.mean(axis=1)
        self.x = np.random.multivariate_normal(mean=self.mean_topk, cov=self.cov, size=self.n)
    def show(self):
        low, high, size = -10, 10, 100
        spacing = np.linspace(low, high, size)
        xx, yy = np.meshgrid(spacing, spacing)
        grid = np.stack((xx, yy), -1)
        Z = self.function(grid).T

        extent = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(Z, cmap=plt.cm.viridis_r, extent=extent, origin='lowerleft')
        losses = self.function(self.x)
        colors = plt.cm.Blues(minmax_scale(losses))
        ax.scatter(self.x.T[0], self.x.T[1], color=colors, s=50)
        ax.scatter(self.topk.T[0], self.topk.T[1], facecolor='none', color='k', s=50)
        plt.show()
    def optimize(self):
        diff=[]
        for step in range(self.max_iter):
            self.old_optimum=self.topk[0]
            self.update(show=True)
            loss=np.linalg.norm(self.old_optimum-self.topk[0])
            self.function(self.topk)
            diff.append(self.function(self.topk[0]))

            # if loss<0.005:
            #     print("brak")
            #     break
            # print(self.topk[0],"diff：",loss)
        print("cma-es:",self.function(self.topk[0]))
        plt.plot(diff,label="cma-es")
        # plt.show()
    def optimize1(self):
        diff=[]
        for step in range(self.max_iter):
            self.old_optimum=self.topk[0]
            self.update1(show=True)
            loss=np.linalg.norm(self.old_optimum-self.topk[0])
            # self.function(self.topk)
            diff.append(self.function(self.topk[0]))
            print("CEM:",self.function(self.topk[0]))
            # if loss<0.005:
            #     print("brak")
            #     break
            # print(self.topk[0],"diff：",loss)
        print("cem:", self.function(self.topk[0]))
        plt.plot(diff,label="cem")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        plt.legend()
        plt.show()
def log_rosenbrock(X, a=1, b=100):
    x,y = X.T
    return np.log((a-x)**2 + b*(y-x**2)**2 + 1.)
def diamond(X):
    x,y = X.T
    return np.sqrt((x+3)**2) + np.sqrt((y-2)**2)
cem=CMAES(diamond,n=200,k=20,mean=[5,5],max_iter=20)
# cem.optimize()
cem.optimize1()