# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:57:33 2023

@author: dpc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:19:05 2023

@author: dpc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:04:27 2023

@author: dpc
"""

import numpy as np
from math import sqrt
from statistics import mean
import os
from gurobipy import *
import pandas as pd
from collections import defaultdict
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(1)
random.seed(1)

def get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up):###生成历史数据，可以在此改参数
    low = (demand_low - mu) / sigma
    up = (demand_up - mu) / sigma  # 计算截断正态分布的函数中用到的上界和下界
    h_demand = stats.truncnorm(low, up, loc=mu, scale=sigma).rvs(T_d).tolist()  #生态分布，随机生成产品的需求信息
    h_w = []
    h_order=[]
    while len(h_w) < T_o:
        w=np.random.rand()
        ww=42+8*w
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
        m.addConstr((sum(P[k] for k in range(index[t1]+1)) >= 1 - h_w[t1] / s), name='c')#9c
        m.addConstr((sum(P[k] for k in range(index[t1])) <= 1 - h_w[t1] / s - epsilon), name='c')#9d
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

def order_worstprofit(c, h_order, h_w, w):
    h_order=h_order.copy()
    h_w=h_w.copy()
    h_order =  sorted(h_order,reverse=True)
    h_w.sort()
    lw = 0
    lw_index=0
    for i in h_w:
        if i > w:
            lw=i
            q=h_order[lw_index]
            break
        lw_index=lw_index+1
    if lw==0:
        q=0
    return (w - c) * q
   
def demand_worstprofit(c, s, w, h_demand, theta):
    h_demand=h_demand.copy()
    h_demand.sort() #对需求升序排列
    location=math.ceil(len(h_demand)*(1-w/s)) 
    xi_e_w = h_demand[location-1] #计算xi_e_w
    temp=0
    T=len(h_demand)
    for j in range(T):
        if h_demand[j] < xi_e_w:
            temp=temp+1/T 
    eta=[1/T for i in range(len(h_demand))]
    eta[h_demand.index(xi_e_w)]=1-w/s-temp
    for v in range(h_demand.index(xi_e_w)+1,len(h_demand)):
        eta[v]=0
    time = range(T)
    m = Model()
    q = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(time, lb=0, vtype=GRB.CONTINUOUS, name="zk")#辅助变量，用于处理目标函数
    m.setObjective((w-c)*q,GRB.MINIMIZE)
    m.addConstr(sum(eta[t]*z[t]*z[t] for t in time) <= theta*theta, name="r")  # 9a
    m.addConstrs((z[t]>=h_demand[t]-q for t in time), name="r1")  # 9b
    m.Params.LogToConsole = 0
    m.optimize()
    if m.Status != 2:
        return [] #模型无解则返回空值
    return (w - c) * q.x

def both_worstprofit(w, s, h_order, c, h_w, theta, xi, eta):   
    h_order=h_order.copy()
    h_w=h_w.copy()
    xi=xi.copy()
    eta=eta.copy()
    xi, eta = zip(*sorted(zip(xi, eta)))#对xi排序
    xi=list(xi)
    eta=list(eta)
    i=1
    while i<=len(xi):
        if sum(eta[:i])>=1-w/s:
            xi_w=xi[i-1]
            break
        i=i+1#计算xi_w
    h_order = sorted(h_order,reverse=True) #h_order降序排列
    h_w.sort()
    lw = 0
    lw_index=0
    for i in h_w:
        if i > w:
            lw=i
            q_s_w=h_order[lw_index]
            break
        lw_index=lw_index+1
    if lw==0:
        q_s_w=0        
    temp=0
    for j in range(len(xi)):
        if xi[j] < xi_w:
            temp=temp+eta[j] #计算根号下括号里最后一项
    eta[xi.index(xi_w)]=1-w/s-temp
    for v in range(xi.index(xi_w)+1,len(xi)):
        eta[v]=0 
    K = range(len(xi))
    m = Model()
    q = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="zk")#辅助变量，用于处理目标函数
    m.setObjective((w-c)*q,GRB.MINIMIZE)
    m.addConstr(sum(eta[k]*z[k]*z[k] for k in K) <= theta*theta, name="r")  # 9a
    m.addConstrs((z[k]>=xi[k]-q for k in K), name="r1")  # 9b
    m.addConstr(q >= q_s_w, name="r")  # 9a
    m.Params.LogToConsole = 0
    m.optimize()
    if m.Status != 2:
        return [] #模型无解则返回空值
    return (w - c) * q.x     

def get_both_radius(theta_list, k, c, s, h_demand, h_order, h_w):
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()
    result_list = []
    xi, eta = get_center(s, h_demand, h_order, h_w)
    for i in range(k):
        el=random.sample(list(range(len(h_demand))), random.choice(list(range(2,len(h_demand)))))                
        train_demand=[]#存放训练集需求数据
        for el1 in el:
            train_demand.append(h_demand[el1])#补充训练集需求数据
        xi_train, eta_train = get_center(s, train_demand, h_order, h_w)
        evaluate_profit=[]
        for b in theta_list:
            max_profit=0
            wholesale=0
            profit_w=[]
            for w in range(c+1, s):
                profit=both_worstprofit(w, s, h_order, c, h_w, b, xi_train, eta_train) 
                profit_w.append(profit)
                if profit > max_profit:
                    max_profit=profit
                    wholesale=w
            evaluate_profit.append(center_profit(s, c, xi, eta, wholesale))
        result_list.append(evaluate_profit)  
    pai=[]
    for d in range(len(theta_list)):
        pai.append(sum(result_list[i][d] for i in range(k))/k)
    max_index = pai.index(max(pai))
    radius=theta_list[max_index]
    print("交叉验证得到的半径",radius)
    return radius

def get_demand_radius(theta_list, k, c, s, h_demand):
    h_demand=h_demand.copy()
    result_list = []
    for i in range(k):
        el=random.sample(list(range(len(h_demand))), random.choice(list(range(2,len(h_demand)))))              
        train_demand=[]#存放训练集需求数据
        for el1 in el:
            train_demand.append(h_demand[el1])#补充训练集需求数据
        evaluate_profit=[]
        for b in theta_list:
            max_profit=0
            wholesale=0
            profit_w=[]
            for w in range(c+1, s):
                profit=demand_worstprofit(c, s, w, train_demand, b)
                profit_w.append(profit)
                if profit > max_profit:
                    max_profit=profit
                    wholesale=w
            evaluate_profit.append(empirical_profit(s, c, h_demand, wholesale))
        result_list.append(evaluate_profit)  
    pai=[]
    for d in range(len(theta_list)):
        pai.append(sum(result_list[i][d] for i in range(k))/k)
    max_index = pai.index(max(pai))
    radius=theta_list[max_index]
    print("交叉验证得到的半径",radius)
    return radius


def experiment(theta_list, k, demand_low, demand_up, c, s, mu, sigma, T_d, T_o, times):  ###列表
    l = 0
    df = pd.DataFrame()
    for r in range(times):
        H_demand, H_order, H_w=get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up)
        h_demand=[]
        h_order=[]
        h_w=[]
        for d in range(T_d):
            h_demand.append(H_demand[d])
        for o in range(T_o):
            h_order.append(H_order[o])
            h_w.append(H_w[o])
        print("历史需求，订单以及批发价格",h_demand, h_order, h_w)
        xi, eta = get_center(s, h_demand, h_order, h_w)
        W=[i for i in range(c+1,s)]
        empirical_w=0
        center_w=0
        demand_w=0
        order_w=0
        both_w=0
        demand_p=0
        empirical_p=0
        center_p=0
        order_p=0
        both_p=0
        R_both=get_both_radius(theta_list, k, c, s, h_demand, h_order, h_w)
        for w in W:
            order_prof=order_worstprofit(c, h_order, h_w, w)
            if order_prof>order_p:
                order_p=order_prof
                order_w=w              
            both_prof=both_worstprofit(w, s, h_order, c, h_w, R_both, xi, eta)
            if both_prof>both_p:
                both_p=both_prof
                both_w=w
        true_profit_or=true_profit(s, c, demand_low, demand_up, mu, sigma, order_w)
        true_profit_bo=true_profit(s, c, demand_low, demand_up, mu, sigma, both_w)
        data = dict()
        data['T_d'] = T_d    
        data['T_o'] = T_o
        data['demand'] = h_demand   
        data['order'] = h_order
        data['w'] = h_w
        data['empirical_w'] = empirical_w
        data['order_w'] = order_w
        data['true_profit_or'] = true_profit_or
        data['both_w'] = both_w    
        data['true_profit_bo'] = true_profit_bo
        df[l] = pd.Series(data)
        l+=1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("试验次数",l)
    or_avg=mean(list(df.loc['true_profit_or']))
    bo_avg=mean(list(df.loc['true_profit_bo']))
    df.at['true_profit_or',times+1]=or_avg
    df.at['true_profit_bo',times+1]=bo_avg
    print(df)
    df.to_csv('订单不确定集large2020.csv')


demand_low=0
demand_up=100
mu=50
sigma=30
T_d=20
T_o=20
c=10
s=50
k=3
times=50
theta_list=[0.1,0.25,0.5,1]
experiment(theta_list, k, demand_low, demand_up, c, s, mu, sigma, T_d, T_o, times)








