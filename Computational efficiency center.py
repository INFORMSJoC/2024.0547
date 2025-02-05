# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:01:42 2023

@author: dpc
"""

from collections import defaultdict
import pandas as pd
import scipy.stats as stats
import numpy as np
import random
from statistics import mean
from gurobipy import *
from math import sqrt
import matplotlib.pyplot as plt
import itertools
from scipy import special
from scipy import optimize
import math
import time
import threading
from scipy.stats import mvn
from scipy.stats import truncnorm
import scipy.integrate as spi
from scipy.stats import multivariate_normal
from minimax_tilting_sampler import TruncatedMVN

np.random.seed(0)
    
def cumulative_distribution(mu, sigma, cor, q, n):###计算累积分布概率
    mu = np.array(mu)###########均值
    sigma=np.array(sigma)####标准差
    covar = cor * sigma[0] * sigma[1]#####协方差
    cov =np.array([[sigma[0] ** 2, covar], [covar, sigma[1] ** 2]])#########协方差矩阵
    lb = [0, 0]#####分布下界
    ub = [100, 100]#########分布上界#########################################
    n_samples = n############抽样个数
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    samples = tmvn.sample(n_samples)
    Demand=samples.T########历史需求数据
    n0, n1 = 0, 0
    for xi in Demand:
        i, j= 0, 1
        if q[i] >= xi[i]:
            n0 += 1
        if q[j] >= xi[j] :
            n1 += 1
    return n0 / n , n1 / n 
    #计算生成的点中有多少个符合条件的，并返回概率
    
def get_H_data(mu, sigma, cor, price, cost, T, demand_low, demand_up):###生成历史数据，可以在此改参数
    mu = np.array(mu)###########均值
    sigma=np.array(sigma)####标准差
    covar = cor * sigma[0] * sigma[1]#####协方差
    cov =np.array([[sigma[0] ** 2, covar], [covar, sigma[1] ** 2]])#########协方差矩阵
    lb = demand_low#####分布下界
    ub = demand_up#########分布上界
    n_samples = T############抽样个数
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    samples = tmvn.sample(n_samples)
    demand=samples.T########历史需求数据
    h_demand=[]
    for j in range(len(demand)):
        temp1=[]
        for i in range(2):
            temp1.append(demand[j][i])
        h_demand.append(temp1)
    order = list()
    wholesale = list()
    tem=[]###存放每次生成的order,避免重复
    while len(order) < T:
        temp = []
        rd0=np.random.rand()
        order_temp0 =lb[0]+(ub[0]-lb[0])*rd0###随机生成历史订单
        rd1=np.random.rand()
        order_temp1 = lb[1]+(ub[1]-lb[1])*rd1###随机生成历史订单
        if order_temp0 not in tem and order_temp1 not in tem:
            tem.append(order_temp0)
            tem.append(order_temp1) 
            temp.append(order_temp0)
            temp.append(order_temp1)
            p = cumulative_distribution(mu, sigma, cor, temp, 100000)##########数点计算概率
            if price[0] * (1 - p[0]) > cost[0] and price[1] * (1 - p[1]) > cost[1]:##排除不合理的订单，保证收入高于成本
                order.append(temp)
                wholesale.append([price[0] * (1 - p[0]), price[1] * (1 - p[1])])###根据订单计算批发价格   
    return h_demand, order, wholesale

def get_center(s, h_demand, h_order, h_w):##LP model single
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()
    T_d=len(h_demand)
    T_o = len(h_order)
    h_order0=[]
    h_order1=[]
    h_w0=[]
    h_w1=[]
    h_demand0=[]
    h_demand1=[]
    for i in range(T_o):
        h_order0.append(h_order[i][0])
        h_order1.append(h_order[i][1])
        h_w0.append(h_w[i][0])
        h_w1.append(h_w[i][1])
    for j in range(T_d):
        h_demand0.append(h_demand[j][0])
        h_demand1.append(h_demand[j][1])
    h_order0.sort()#对历史订单排序
    h_order1.sort()#对历史订单排序
    h_w0=sorted(h_w0,reverse=True)
    h_w1=sorted(h_w1,reverse=True)
    epsilon = 0.00000001
    support0=[]
    support1=[]
    support0=h_demand0+h_order0
    support1=h_demand1+h_order1
    support=[]
    for t in range(len(support0)):
        for j in range(len(support1)):
            if support0[t] not in h_demand0 or support1[j] not in h_demand1:
                support.append([support0[t],support1[j]])
    for i in range(T_d):
        support.append(h_demand[i])
    q0_index=[[] for i in range(T_o)]
    q1_index=[[] for i in range(T_o)]
    q00_index=[[] for i in range(T_o)]
    q11_index=[[] for i in range(T_o)]
    for b in range(T_o):
        index0=0
        for a in range(len(support)):
            if support[a][0]<=h_order0[b]:
                q0_index[b].append(index0)
            index0=index0+1
    for b in range(T_o):
        index0=0
        for a in range(len(support)):
            if support[a][0] < h_order0[b]:
                q00_index[b].append(index0)
            index0=index0+1
    for d in range(T_o):
        index1=0
        for e in range(len(support)):
            if support[e][1]<=h_order1[d]:
                q1_index[d].append(index1)
            index1=index1+1
    for d in range(T_o):
        index1=0
        for e in range(len(support)):
            if support[e][1]<h_order1[d]:
                q11_index[d].append(index1)
            index1=index1+1
    K = range(len(support))
    tim=range(T_d)
    m = Model()
    P = m.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="p")
    pai = m.addVars(tim, K, lb=0, vtype=GRB.CONTINUOUS, name="pai")#辅助变量，用于处理目标函数
    m.setObjective(sum(((h_demand[t][0]-support[k][0])**2+(h_demand[t][1]-support[k][1])**2)\
                       *pai[t,k] for k in K for t in tim), GRB.MINIMIZE)
    m.addConstrs((sum(pai[t, k] for k in K) == 1 / T_d for t in tim), name="r")  # 9a
    m.addConstrs((sum(pai[t, k] for t in tim) == P[k] for k in K), name="r1")  # 9b
    for t1 in range(T_o):  
        m.addConstr((sum(P[k] for k in q0_index[t1]) >= 1 - h_w0[t1] / s[0]), name='c')#9c
        m.addConstr((sum(P[k] for k in q00_index[t1]) <= 1 - h_w0[t1] / s[0] - epsilon), name='c')#9d
        m.addConstr((sum(P[k] for k in q1_index[t1]) >= 1 - h_w1[t1] / s[1]), name='c')#9c
        m.addConstr((sum(P[k] for k in q11_index[t1]) <= 1 - h_w1[t1] / s[1] - epsilon), name='c')#9d
    # m.Params.LogToConsole = 0
    m.Params.TimeLimit=3600
    # m.setParam(GRB.Param.LogFile, "gurobi1.log"+str(T)) 
    # m.optimize()
    m.optimize()
    if m.Status != 2:
        return [], [] #模型无解则返回空值
    xi=[]
    eta=[]
    for v in range(len(support)):
        if P[v].x >= 0.0000001:
            xi.append(support[v])
            eta.append(P[v].x)
    # print('球心分布1',m.objVal,xi,eta)
    return xi, eta

def get_center_2(price, h_demand, demand_low, demand_up, h_order, h_w):###计算球心的分布
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()
    T = len(h_order)
    h_order0=[]
    h_order1=[]
    h_w0=[]
    h_w1=[]
    h_demand0=[]
    h_demand1=[]
    for i in range(T):
        h_order0.append(h_order[i][0])
        h_order1.append(h_order[i][1])
        h_w0.append(h_w[i][0])
        h_w1.append(h_w[i][1])
    for j in range(T):
        h_demand0.append(h_demand[j][0])
        h_demand1.append(h_demand[j][1])
    h_order0.sort()#对历史订单排序
    h_order1.sort()#对历史订单排序
    h_w0=sorted(h_w0,reverse=True)
    h_w1=sorted(h_w1,reverse=True)
    epsilon = 0.00000001
    data0=h_demand0+h_order0+[demand_low[0]]+[demand_up[0]]
    data1=h_demand1+h_order1+[demand_low[1]]+[demand_up[1]]
    data0.sort()
    data1.sort()
    # print('data0',data0)
    # print('data1',data1)
    time = range(T)
    Set = range((2*T + 1)*(2*T + 1))
    supportpoint = []
    Z = []
    for t in time:
        support=[]
        print('111111111111111')
        for s0 in range(2*T+1):
            for s1 in range(2*T+1):
                temp=[0,0]
                if h_demand0[t]>=data0[s0+1]:
                    temp[0]=data0[s0+1]
                if h_demand0[t]<=data0[s0]:
                    temp[0]=data0[s0]+epsilon
                if h_demand1[t]>=data1[s1+1]:
                    temp[1]=data1[s1+1]
                if h_demand1[t]<=data1[s1]:
                    temp[1]=data1[s1]+epsilon
                support.append(temp)
        supportpoint.append(support)
    # print('supportpoint',supportpoint)
    
    for t in time:
        zzz=[]
        print('22222222222222')
        for s0 in range(2*T+1):
            for s1 in range(2*T+1):
                tem=[0,0]
                if h_order0[t]>=data0[s0+1]:
                    tem[0]=1
                if h_order1[t]>=data1[s1+1]:
                    tem[1]=1
                zzz.append(tem)
        Z.append(zzz)
    # print('Z',Z)
    m = Model()
    Beta = m.addVars(time, Set, lb=0, vtype=GRB.CONTINUOUS, name="Beta")
    m.setObjective(sum(Beta[t,s]*((h_demand[t][0]-supportpoint[t][s][0])*(h_demand[t][0]-supportpoint[t][s][0]) \
                       + (h_demand[t][1]-supportpoint[t][s][1])*(h_demand[t][1]-supportpoint[t][s][1])) for s in Set for t in time), GRB.MINIMIZE)
    m.addConstrs((sum(Beta[t,s] for s in Set) == 1 / T for t in time), name="r")  # 9b
    for t1 in time:  
        m.addConstr((sum(Beta[t,s]*Z[t1][s][0] for s in Set for t in time) == 1 - h_w0[t1] / price[0]), name='c' + str(t1))#9c
    for t1 in time:  
        m.addConstr((sum(Beta[t,s]*Z[t1][s][1] for s in Set for t in time) == 1 - h_w1[t1] / price[1]), name='c' + str(t1))#9c
    #m.Params.LogToConsole = 0
    #m.write('dddddddddddddd.lp')
    m.Params.TimeLimit=3600
    m.optimize()
    if m.Status != 2:
        return [] #模型无解则返回空值
    xi=[]
    eta=[]
    for t in time:
        for s in Set:
            if Beta[t,s].x >= 0.0000001:
                xi.append(supportpoint[t][s])
                eta.append(Beta[t,s].x)
    # print('球心分布1',m.objVal,xi,eta)
    return xi, eta

def get_z_and_position(order, demand_low, demand_up, T):
    order=order.copy()
    S=range((T+1)*(T+1))
    order_x=[]
    order_y=[]
    order_x.append(demand_low[0])
    order_y.append(demand_low[1])
    for j in range(T):
        order_x.append(order[j][0])
        order_y.append(order[j][1])
    order_x.append(demand_up[0])
    order_y.append(demand_up[1])
    order_x.sort()
    order_y.sort()
    position=[]
    for i in S:
        position_y=i//(T+1)#####从0开始计行数
        position_x=i%(T+1)#####从0开始计行数
        position.append([position_y,position_x])
    z= [[[],[]]]*len(S)
    for s in S:
        z[s] = [[], []]
        for k in range(T):
            if order_x[position[s][1]] >= order[k][0]:
                z[s][0].append(0)
            else:
                z[s][0].append(1)
            if order_y[position[s][0]] >= order[k][1]:
                z[s][1].append(0)
            else:
                z[s][1].append(1)
    return position, z, order_x, order_y

def get_data_driven(demand, order, wholesale, T, price, demand_low, demand_up):
    order=order.copy()
    S=range((T+1)*(T+1))
    position, z, order_x, order_y=get_z_and_position(order, demand_low, demand_up, T)
    time = range(T)
    epsilon = 0.0000001     
    m = Model()
    Beta = m.addVars(S, time, lb=0, vtype=GRB.CONTINUOUS, name="Beta")
    P = m.addVars(S, time, 2, vtype=GRB.CONTINUOUS, name="p")
    Z = m.addVars(S, time, vtype=GRB.CONTINUOUS, name="z")
    X = m.addVars(S, time, 2, vtype=GRB.CONTINUOUS, name="x")
    m.setObjective(sum(Z[s, t] for s in S for t in time), GRB.MINIMIZE)
    m.addConstrs((sum(Beta[s, t] for s in S) == 1 / T for t in time), name="c1")
    m.addConstrs((sum(Beta[s, t] * z[s][i][t1] for s in S for t in time) == (price[i] - wholesale[t1][i])/price[i] for i in range(2) for t1 in time), "st3")
    for s in S:
        m.addConstrs((P[s, k, 1] >= Beta[s, k] * order_y[position[s][0]]+epsilon for k in time),name='s0')
        m.addConstrs((P[s, k, 0] >= Beta[s, k] * order_x[position[s][1]]+epsilon for k in time),name='q1')
        m.addConstrs((P[s, k, 1] <= Beta[s, k] * order_y[position[s][0]+1] for k in time),name='s0')
        m.addConstrs((P[s, k, 0] <= Beta[s, k] * order_x[position[s][1]+1] for k in time),name='q1')
    m.addConstrs((X[s, t, 0] >= P[s, t, 0] - Beta[s, t] * demand[t][0] for s in S for t in time), name="e")
    m.addConstrs((X[s, t, 1] >= P[s, t, 1] - Beta[s, t] * demand[t][1] for s in S for t in time), name="e")
    m.addConstrs((X[s, t, 0] >= Beta[s, t] * demand[t][0] - P[s, t, 0] for s in S for t in time), name="e")
    m.addConstrs((X[s, t, 1] >= Beta[s, t] * demand[t][1] - P[s, t, 1] for s in S for t in time), name="e")
    m.addConstrs((Beta[s, t] * Z[s, t] >= X[s, t, 0] * X[s, t, 0] + X[s, t, 1] * X[s, t, 1] for s in S for t in time), name='ss')
    # m.Params.Presolve = 0
    m.Params.TimeLimit=3600
    # m.Params.LogToConsole = 0
    # m.write('sssssssssssssss.lp')
    m.optimize()
    # print("解球心的模型status",m.Status)
    xi=[]
    eta=[]
    if m.Status == 2:
        d = defaultdict(int)
        for s in S:
            for t in time:
                if Beta[s, t].x > 0.001:
                    d[(round(P[s, t, 0].x / Beta[s, t].x,8), round(P[s, t, 1].x / Beta[s, t].x, 8))] += round(Beta[s, t].x,8)
        eta = list(d.values())
        xi = list(d.keys())
    # print('球心分布2',m.objVal,xi,eta)
    return xi, eta

demand_low=[0,0]
demand_up=[100,100]
mu=[50,50]
sigma=[30,30]
cor=0.3
c=[10,10]
price=[100,100]
time_record1=[]
time_record2=[]
time_record3=[]

for R in range(1):
    T=60+R*10
    print('TTTTTTTTTTTTTT',T)
    h_demand, h_order, h_w=get_H_data(mu, sigma, cor, price, c, T, demand_low, demand_up)
    print('历史数据',h_demand,h_order,h_w)
    # start_time1=time.time()
    # get_center(price, h_demand, h_order, h_w)
    # end_time1=time.time() 
    # print('总时间',end_time1-start_time1) 
    # time_record1.append(end_time1-start_time1)
    start_time2=time.time()
    get_center_2(price, h_demand, demand_low, demand_up, h_order, h_w)
    end_time2=time.time() 
    print('总时间',end_time2-start_time2) 
    time_record2.append(end_time2-start_time2)
    # start_time3=time.time()
    # get_data_driven(h_demand, h_order, h_w, T, price, demand_low, demand_up)
    # end_time3=time.time() 
    # print('总时间',end_time3-start_time3) 
    # time_record3.append(end_time3-start_time3)

print('time_record1',time_record1)
print('time_record2',time_record2)
print('time_record2',time_record3)



