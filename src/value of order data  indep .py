# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 08:54:56 2023

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

np.random.seed(0)
random.seed(0)

def get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up):###生成历史数据.列表
    h_demand=[[],[]]
    h_order=[[],[]]
    h_w=[[],[]]
    for i in range(len(mu)):        
        low = (demand_low[i] - mu[i]) / sigma[i]
        up = (demand_up[i] - mu[i]) / sigma[i]  # 计算截断正态分布的函数中用到的上界和下
        h_demand[i] = stats.truncnorm(low, up, loc=mu[i], scale=sigma[i]).rvs(T_d).tolist()  #生态分布，随机生成产品的需求信息
        while len(h_w[i]) < T_o:
            w=np.random.rand()
            ww=c[i]+(s[i]-c[i])*w
            order=stats.truncnorm.ppf(1-ww/s[i], low, up, loc=mu[i], scale=sigma[i])
            if ww not in h_w[i]:
                h_w[i].append(ww)
                h_order[i].append(order)
        h_w[i] = list(h_w[i])
        h_order[i]=list(h_order[i])
        # print('历史需求数据',h_demand)
        # print('历史订单数据',h_order)
        # print('历史价格数据',h_w)
    return h_demand,h_order,h_w

def generate_W(s, c):#####生成多个产品需要迭代的w
    step = 2 #################################################
    r_ = [range(c[i]+1,s[i],step) for i in range(len(c))]
    w = []
    w_ = []
    def generate_w(layer):
        if layer>=len(r_):
            return
        for number in r_[layer]:
            w_.append(number)
            generate_w(layer+1)
            if layer == len(r_)-1:
                w.append(w_[:])
            w_.pop()
    generate_w(0)
    W=w
    return W

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


def both_worstprofit(w, s, h_order, c, h_w, theta, xi, eta):   
    h_order=h_order.copy()
    h_w=h_w.copy()
    xi=xi.copy()
    eta=eta.copy()
    xi_w=[0,0]
    q_s_w=[0,0]
    for j in range(len(c)):
        xi[j],eta[j]=zip(*sorted(zip(xi[j], eta[j])))
        xi[j]=list(xi[j])
        eta[j]=list(eta[j])
        i=1
        while i<=len(xi[j]):
            if sum(eta[j][:i])>=1-w[j]/s[j]:
                xi_w[j]=xi[j][i-1]
                break
            i=i+1#计算xi_w
        h_order[j] = sorted(h_order[j],reverse=True) #h_order降序排列
        h_w[j].sort()
        lw = 0
        lw_index=0
        for a in h_w[j]:
            if a > w[j]:
                lw=a
                q_s_w[j]=h_order[j][lw_index]
                break
            lw_index=lw_index+1
        if lw==0:
            q_s_w[j]=0        
        temp=0
        for l in range(len(xi[j])):
            if xi[j][l] < xi_w[j]:
                temp=temp+eta[j][l] #计算根号下括号里最后一项
        eta[j][xi[j].index(xi_w[j])]=1-w[j]/s[j]-temp
        for v in range(xi[j].index(xi_w[j])+1,len(xi[j])):
            eta[j][v]=0  
    K0 = range(len(xi[0]))
    K1 = range(len(xi[1]))
    m = Model()
    q = m.addVars(len(c), lb=0, vtype=GRB.CONTINUOUS, name="q")
    z0 = m.addVars(K0, lb=0, vtype=GRB.CONTINUOUS, name="z0")#辅助变量，用于处理目标函数
    z1 = m.addVars(K1, lb=0, vtype=GRB.CONTINUOUS, name="z1")#辅助变量，用于处理目标函数
    m.setObjective(sum((w[i]-c[i])*q[i] for i in range(len(c))),GRB.MINIMIZE)
    m.addConstr(sum(eta[0][k]*z0[k]*z0[k] for k in K0)+sum(eta[1][k]*z1[k]*z1[k] for k in K1) <= theta*theta, name="r")  # 9a
    m.addConstrs((z0[k]>=xi[0][k]-q[0] for k in K0), name="r1")  # 9b
    m.addConstrs((z1[k]>=xi[1][k]-q[1] for k in K1), name="r1")  # 9b
    m.addConstrs((q[i] >= q_s_w[i] for i in range(len(c))), name="r")  # 9a
    m.Params.LogToConsole = 0
    m.optimize()
    if m.Status != 2:
        return [] #模型无解则返回空值
    return sum((w[i]-c[i])*q[i].x for i in range(len(c))) 


def demand_worstprofit(c, s, w, h_demand, theta):
    h_demand=h_demand.copy()
    h_demand[0].sort() #对需求升序排列
    h_demand[1].sort() #对需求升序排列
    xi=[[],[]]
    eta=[[],[]]
    xi_e_w=[0,0]
    T=len(h_demand[0])
    eta[0]=[1/T for i in range(T)]
    xi[0]=h_demand[0]
    eta[1]=[1/T for i in range(T)]
    xi[1]=h_demand[1]
    for j in range(len(c)):
        location=math.ceil(len(h_demand[j])*(1-w[j]/s[j])) 
        xi_e_w[j] = h_demand[j][location-1] #计算xi_e_w
        temp=0
        for i in range(T):
            if h_demand[j][i] < xi_e_w[j]:
                temp=temp+1/T #计算根号下括号里最后一项
        eta[j][h_demand[j].index(xi_e_w[j])]=1-w[j]/s[j]-temp
        for v in range(h_demand[j].index(xi_e_w[j])+1,len(h_demand[j])):
            eta[j][v]=0
    time = range(T)
    m = Model()
    q = m.addVars(2,lb=0, vtype=GRB.CONTINUOUS, name="q")
    z = m.addVars(2,time, lb=0, vtype=GRB.CONTINUOUS, name="zk")#辅助变量，用于处理目标函数
    m.setObjective(sum((w[i]-c[i])*q[i] for i in range(len(c))),GRB.MINIMIZE)
    m.addConstr(sum(eta[i][t]*z[i,t]*z[i,t] for t in time for i in range(2)) <= theta*theta, name="r")  # 9a
    m.addConstrs((z[i,t]>=xi[i][t]-q[i] for t in time for i in range(len(c))), name="r1")  # 9b
    m.addConstrs((q[i] >= 0 for i in range(len(c))), name="r")
    m.Params.LogToConsole = 0
    m.optimize()
    if m.Status != 2:
        return [] #模型无解则返回空值
    return sum((w[i]-c[i])*q[i].x for i in range(len(c)))

def get_both_radius(theta_list, k, c, s, h_demand, h_order, h_w):
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()
    result_list = []
    W=generate_W(s, c)
    xi=[[],[]]
    eta=[[],[]]
    for ii in range(len(c)):
            xi[ii], eta[ii] = get_center(s[ii], h_demand[ii], h_order[ii], h_w[ii])
    for j in range(k):
        el=random.sample(list(range(len(h_demand[0]))), random.choice(list(range(2,len(h_demand[0])))))                
        train_demand=[[],[]]#存放训练集需求数据
        for aa in range(len(c)):
            for el1 in el:
                train_demand[aa].append(h_demand[aa][el1])#补充训练集需求数据   
        xi_train=[[],[]]
        eta_train=[[],[]]  
        for bb in range(len(c)):
            xi_train[bb], eta_train[bb] = get_center(s[bb], train_demand[bb], h_order[bb], h_w[bb])
        evaluate_profit=[]
        for b in theta_list:
            max_profit=0
            wholesale=[0,0]
            profit_w=[]
            for w in W:
                profit=both_worstprofit(w, s, h_order, c, h_w, b, xi_train, eta_train) 
                profit_w.append(profit)
                if profit > max_profit:
                    max_profit=profit
                    wholesale[0]=w[0]
                    wholesale[1]=w[1]
            evaluate_profit.append(sum(center_profit(s[i], c[i], xi[i], eta[i], wholesale[i]) for i in range(2)))
        result_list.append(evaluate_profit)  
    pai=[]
    for d in range(len(theta_list)):
        pai.append(sum(result_list[i][d] for i in range(k))/k)
    max_index = pai.index(max(pai))
    radius=theta_list[max_index]
    return radius

def get_demand_radius(theta_list, k, c, s, h_demand):
    h_demand=h_demand.copy()
    result_list = []
    W=generate_W(s, c)
    for i in range(k):     
        el=random.sample(list(range(len(h_demand[0]))), random.choice(list(range(2,len(h_demand[0]))))) 
        train_demand=[[],[]]#存放训练集需求数据
        for aa in range(len(c)):
            for el1 in el:
                train_demand[aa].append(h_demand[aa][el1])#补充训练集需求数据     
        evaluate_profit=[]
        for b in theta_list:
            max_profit=0
            wholesale=[0,0]
            profit_w=[]
            for w in W:
                profit=demand_worstprofit(c, s, w, train_demand, b)
                profit_w.append(profit)
                if profit > max_profit:
                    max_profit=profit
                    wholesale[0]=w[0]
                    wholesale[1]=w[1]
            evaluate_profit.append(sum(empirical_profit(s[i], c[i], h_demand[i], wholesale[i]) for i in range(2)))
        result_list.append(evaluate_profit)  
    pai=[]
    for d in range(len(theta_list)):
        pai.append(sum(result_list[i][d] for i in range(k))/k)
    max_index = pai.index(max(pai))
    radius=theta_list[max_index]
    return radius

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
   


def experiment(theta_list, k, demand_low, demand_up, c, s, mu, sigma, T_d, T_o, times):  ###列表
    l = 0
    df = pd.DataFrame()
    for r in range(times):
        H_demand, H_order, H_w=get_H_data(mu, sigma, s, c, T_d, T_o, demand_low, demand_up)
        h_demand=[[],[]]
        h_order=[[],[]]
        h_w=[[],[]]
        for d in range(T_d):
            h_demand[0].append(H_demand[0][d])
            h_demand[1].append(H_demand[1][d])
        for o in range(T_o):
            h_order[0].append(H_order[0][o])
            h_order[1].append(H_order[1][o])
            h_w[0].append(H_w[0][o])
            h_w[1].append(H_w[1][o])
        print("历史需求，订单以及批发价格",h_demand, h_order, h_w)
        xi=[[],[]]
        eta=[[],[]]
        for i in range(len(c)):
            xi[i],eta[i]=get_center(s[i], h_demand[i], h_order[i], h_w[i])
        W=generate_W(s, c)
        # empirical_w=[0,0]
        # center_w=[0,0]
        # demand_w=[0,0]
        # order_w=[0,0]
        both_w=[0,0]
        # demand_p=0
        # empirical_p=0
        # center_p=0
        # order_p=0
        both_p=0
        # R_demand=get_demand_radius(theta_list, k, c, s, h_demand)
        R_both=get_both_radius(theta_list, k, c, s, h_demand, h_order, h_w)
        for w in W:
            # empirical_prof=sum(empirical_profit(s[i], c[i], h_demand[i], w[i]) for i in range(2))
            # if empirical_prof > empirical_p:
            #     empirical_p=empirical_prof
            #     empirical_w[0]=w[0]
            #     empirical_w[1]=w[1]
            # center_prof=sum(center_profit(s[i], c[i], xi[i], eta[i], w[i]) for i in range(2))
            # if center_prof > center_p:
            #     center_p=center_prof
            #     center_w[0]=w[0]
            #     center_w[1]=w[1]
            # order_prof=sum(order_worstprofit(c[i], h_order[i], h_w[i], w[i]) for i in range(2))
            # if order_prof>order_p:
            #     order_p=order_prof
            #     order_w[0]=w[0]
            #     order_w[1]=w[1]
            # demand_prof=demand_worstprofit(c, s, w, h_demand, R_demand)
            # if demand_prof>demand_p:
            #     demand_p=demand_prof
            #     demand_w[0]=w[0]  
            #     demand_w[1]=w[1] 
            both_prof=both_worstprofit(w, s, h_order, c, h_w, R_both, xi, eta)
            if both_prof>both_p:
                both_p=both_prof
                both_w[0]=w[0]
                both_w[1]=w[1]
        # true_profit_em=sum(true_profit(s[i], c[i], demand_low[i], demand_up[i], mu[i], sigma[i], empirical_w[i]) for i in range(2))
        # true_profit_ce=sum(true_profit(s[i], c[i], demand_low[i], demand_up[i], mu[i], sigma[i], center_w[i]) for i in range(2))
        # true_profit_or=sum(true_profit(s[i], c[i], demand_low[i], demand_up[i], mu[i], sigma[i], order_w[i]) for i in range(2))
        # true_profit_de=sum(true_profit(s[i], c[i], demand_low[i], demand_up[i], mu[i], sigma[i], demand_w[i]) for i in range(2))
        true_profit_bo=sum(true_profit(s[i], c[i], demand_low[i], demand_up[i], mu[i], sigma[i], both_w[i]) for i in range(2))
        data = dict()
        data['T_d'] = T_d
        data['T_o'] = T_o
        for j in range(len(h_demand)):
            data['demand'+str(j)] = h_demand[j]
        for j in range(len(h_order)):
            data['order'+str(j)] = h_order[j]
        for j in range(len(h_w)):
            data['w'+str(j)] = h_w[j]
        data['both_theta'] = R_both
        # data['empirical_w'] = empirical_w
        # data['true_profit_em'] = true_profit_em
        # data['center_w'] = center_w
        # data['true_profit_ce'] = true_profit_ce
        # data['demand_w'] = demand_w
        # data['true_profit_de'] = true_profit_de
        # data['order_w'] = order_w
        # data['true_profit_or'] = true_profit_or
        data['both_w'] = both_w
        data['true_profit_bo'] = true_profit_bo
        df[l] = pd.Series(data)
        l+=1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("试验次数",l)
    # em_avg=mean(list(df.loc['true_profit_em']))
    # em_var=np.var(list(df.loc['true_profit_em']))
    # ce_avg=mean(list(df.loc['true_profit_ce']))
    # ce_var=np.var(list(df.loc['true_profit_ce']))
    # de_avg=mean(list(df.loc['true_profit_de']))
    # de_var=np.var(list(df.loc['true_profit_de']))
    # or_avg=mean(list(df.loc['true_profit_or']))
    # or_var=np.var(list(df.loc['true_profit_or']))
    bo_avg=mean(list(df.loc['true_profit_bo']))
    bo_var=np.var(list(df.loc['true_profit_bo']))
    # df.at['true_profit_em',times+1]=em_avg
    # df.at['true_profit_em',times+2]=em_var
    # df.at['true_profit_ce',times+1]=ce_avg
    # df.at['true_profit_ce',times+2]=ce_var
    # df.at['true_profit_de',times+1]=de_avg
    # df.at['true_profit_de',times+2]=de_var
    # df.at['true_profit_or',times+1]=or_avg
    # df.at['true_profit_or',times+2]=or_var
    df.at['true_profit_bo',times+1]=bo_avg
    df.at['true_profit_bo',times+2]=bo_var
    print(df)
    df.to_csv('独立3-50.csv')

demand_low=[0,0]
demand_up=[100,100]
mu=[50,50]
sigma=[30,20]
T_d=3
T_o=50
c=[10,10]
s=[40,50]
times=100
k=5
theta_list=[0.5,1,2,3]
experiment(theta_list, k, demand_low, demand_up, c, s, mu, sigma, T_d, T_o, times)









