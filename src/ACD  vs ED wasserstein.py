# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:14:03 2023

@author: dpc
"""

import numpy as np
from math import sqrt
import statistics
from statistics import mean
import os
from gurobipy import *
import pandas as pd
from collections import defaultdict
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import integrate
import scipy.stats as stats
np.random.seed(0)
random.seed(0)

def get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up):###生成历史数据，可以在此改参数
    low = (demand_low - mu) / sigma
    up = (demand_up - mu) / sigma  # 计算截断正态分布的函数中用到的上界和下界
    h_demand = stats.truncnorm(low, up, loc=mu, scale=sigma).rvs(T_d).tolist()  #生态分布，随机生成产品的需求信息
    h_w = []
    h_order=[]
    while len(h_w) < T_o:
        w=np.random.rand()
        ww=c+40*w
        order=stats.truncnorm.ppf(1-ww/s, low, up, loc=mu, scale=sigma)
        if ww not in h_w:
            h_w.append(ww)
            h_order.append(order)
    h_w = list(h_w)
    h_order=list(h_order)
    return h_demand,h_order,h_w

def get_center(s, h_demand, h_order, h_w):##LP model single
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()
    T_o=len(h_order)
    T_d=len(h_demand)
    h_order.sort()#对历史订单排序
    h_w=sorted(h_w,reverse=True)
    epsilon = 0.0000001
    support=h_demand+h_order
    support.sort()
    index=[]
    for u in range(T_o):
        index.append(support.index(h_order[u]))###获取支撑里面订单点对应的索引
    K = range(len(support))
    time=range(T_d)
    m = Model()
    P = m.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="p")
    pai = m.addVars(time, K, lb=0, vtype=GRB.CONTINUOUS, name="pai")#辅助变量，用于处理目标函数
    m.setObjective(sum((h_demand[t]-support[k])*(h_demand[t]-support[k])*pai[t,k] for k in K for t in time), GRB.MINIMIZE)
    m.addConstrs((sum(pai[t, k] for k in K) == 1 / T_d for t in time), name="r")  # 9a
    m.addConstrs((sum(pai[t, k] for t in time) == P[k] for k in K), name="r1")  # 9b
    for t1 in range(T_o):  
        m.addConstr((sum(P[k] for k in range(index[t1]+1)) >= 1 - h_w[t1] / s-0.02), name='c')#9c
        m.addConstr((sum(P[k] for k in range(index[t1])) <= 1 - h_w[t1] / s - epsilon+0.02), name='c')#9d
    m.Params.LogToConsole = 0
    m.optimize()
    if m.Status != 2:
        return [], [] #模型无解则返回空值
    xi=[]
    eta=[]
    for v in range(len(support)):
        if P[v].x >= 0.0000001:
            xi.append(support[v])
            eta.append(P[v].x)
    return xi, eta
    
def true_profit(s, c, demand_low, demand_up, mu, sigma, w):   
    low = (demand_low - mu) / sigma
    up = (demand_up - mu) / sigma  # 计算截断正态分布的函数中用到的上界和下界
    q=stats.truncnorm.ppf(1-w/s, low, up, loc=mu, scale=sigma)#根据FOC计算真实分布下的q
    return (w - c) * q

def empirical_profit(s, c, h_demand, w):
    h_demand=h_demand.copy()
    h_demand.sort() #对需求升序排列
    location=math.ceil(len(h_demand)*(1-w/s)) #确定需要累积的需求个数
    q = h_demand[location-1] #根据FOC计算q
    return (w - c) * q #返回收益

def center_profit(s, c, xi, eta, w):
    xi=xi.copy()
    eta=eta.copy()
    xi, eta = zip(*sorted(zip(xi, eta)))#对xi排序
    i=1
    while i<=len(xi):
        if sum(eta[:i])>=1-w/s:
            xi_w=xi[i-1]
            break
        i=i+1#计算xi_w
    return (w - c) * xi_w #返回收益

# low=0
# up=200
# mu=50
# sigma=20    
# distribution=[5, 10]
# probabilities = [0.4, 0.6]
def integrand(x):

    p=0
    for j in range(2):
        v=0
        p=p+probabilities[j]
        if p>=x:
            v=distribution[j]
            break
        
    truncnorm_ppf=stats.truncnorm.ppf(x, (demand_low-mu)/sigma, (demand_up-mu)/sigma, loc=mu, scale=sigma)
    return (truncnorm_ppf-v)**2

# integral, _ = quad(integrand, 0, 1)
# print(integral)

def experiment(demand_low, demand_up, c, s, mu, sigma, T_o, TT, times):  ###列表
    Ave1=[]
    Ave=[]
    for T_d in TT:
        average1=[]
        average=[]
        for r in range(times):
            h_demand, h_order, h_w=get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up)
            print("历史需求，订单以及批发价格",h_demand, h_order, h_w)
            xi, eta = get_center(s, h_demand, h_order, h_w)
            print('中心分布',xi, eta)
            h_demand=sorted(h_demand)
            distribution1=h_demand
            print("经验分布",distribution1)
            probabilities1=[1/T_d for i in range(T_d)]
            def integrand1(x):
                p1=0
                for j in range(T_d):
                    v1=0
                    p1=p1+probabilities1[j]
                    if p1>=x:
                        v1=distribution1[j]
                        break            
                truncnorm_ppf=stats.truncnorm.ppf(x, (demand_low-mu)/sigma, (demand_up-mu)/sigma, loc=mu, scale=sigma)
                return (truncnorm_ppf-v1)**2
            integral1, _ = quad(integrand1, 0, 1)
            print('经验WD',math.sqrt(integral1))
            distribution=xi
            probabilities=eta
            def integrand(x):
                p=0
                for j in range(len(xi)):
                    v=0
                    p=p+probabilities[j]
                    if p>=x:
                        v=distribution[j]
                        break            
                truncnorm_ppf=stats.truncnorm.ppf(x, (demand_low-mu)/sigma, (demand_up-mu)/sigma, loc=mu, scale=sigma)
                return (truncnorm_ppf-v)**2
            integral, _ = quad(integrand, 0, 1)
            print('中心WD',math.sqrt(integral))
            average1.append(math.sqrt(integral1))
            average.append(math.sqrt(integral))
        Ave1.append(round(statistics.mean(average1),2))
        Ave.append(round(statistics.mean(average),2))
    print('经验距离',Ave1)
    print('中心距离',Ave)

demand_low=0
demand_up=100
mu=50
sigma=20
TT=[5,10,15,20,25,30]
T_o=5
c=10
s=50
times=20
experiment(demand_low, demand_up, c, s, mu, sigma, T_o, TT, times)








