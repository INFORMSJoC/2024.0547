# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:45:00 2023

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
from scipy.stats import mvn
from scipy.stats import truncnorm
import scipy.integrate as spi
from scipy.stats import multivariate_normal
from minimax_tilting_sampler import TruncatedMVN
np.random.seed(0)
random.seed(0)
        
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
    
def get_H_data(mu, sigma, cor, price, cost, T_d, T_o, demand_low, demand_up):###生成历史数据，可以在此改参数
    mu = np.array(mu)###########均值
    sigma=np.array(sigma)####标准差
    covar = cor * sigma[0] * sigma[1]#####协方差
    cov =np.array([[sigma[0] ** 2, covar], [covar, sigma[1] ** 2]])#########协方差矩阵
    lb = demand_low#####分布下界
    ub = demand_up#########分布上界
    n_samples = T_d############抽样个数
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
    while len(order) < T_o:
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
    time=range(T_d)
    m = Model()
    P = m.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="p")
    pai = m.addVars(time, K, lb=0, vtype=GRB.CONTINUOUS, name="pai")#辅助变量，用于处理目标函数
    m.setObjective(sum(((h_demand[t][0]-support[k][0])**2+(h_demand[t][1]-support[k][1])**2)\
                       *pai[t,k] for k in K for t in time), GRB.MINIMIZE)
    m.addConstrs((sum(pai[t, k] for k in K) == 1 / T_d for t in time), name="r")  # 9a
    m.addConstrs((sum(pai[t, k] for t in time) == P[k] for k in K), name="r1")  # 9b
    for t1 in range(T_o):  
        m.addConstr((sum(P[k] for k in q0_index[t1]) >= 1 - h_w0[t1] / s[0]), name='c')#9c
        m.addConstr((sum(P[k] for k in q00_index[t1]) <= 1 - h_w0[t1] / s[0] - epsilon), name='c')#9d
        m.addConstr((sum(P[k] for k in q1_index[t1]) >= 1 - h_w1[t1] / s[1]), name='c')#9c
        m.addConstr((sum(P[k] for k in q11_index[t1]) <= 1 - h_w1[t1] / s[1] - epsilon), name='c')#9d
    m.Params.LogToConsole = 0
    m.optimize()
    # print('--------modelStatus', m.Status)
    if m.Status != 2:
        return [], [] #模型无解则返回空值
    xi=[]
    eta=[]
    for v in range(len(support)):
        if P[v].x >= 0.0000001:
            xi.append(support[v])
            eta.append(P[v].x)
    return xi, eta

def true_profit(mu, sigma, cor, w, s, c): 
    left0=0 #初始左端点
    right0= 100 #初始右端点
    while left0<=right0-0.001: #判断条件
        mid0 = (left0+right0)/2  #中间点         
        p0,pp = cumulative_distribution(mu, sigma, cor, [mid0,100], 100000)
        if p0==1-w[0]/s[0]:
            q0=p0
            break
        if p0>1-w[0]/s[0]:
            right0=mid0
        else:
            left0=mid0
    q0=(left0+right0)/2
    left1=0 #初始左端点
    right1= demand_up[1] #初始右端点
    while left1<=right1-0.001: #判断条件
        mid1= (left1+right1)/2  #中间点          
        pa,p1 = cumulative_distribution(mu, sigma, cor, [100,mid1], 100000)
        if p1==1-w[1]/s[1]:
            q1=p1
            break
        if p1>1-w[1]/s[1]:
            right1=mid1
        else:
            left1=mid1
    q1=(left1+right1)/2 
    return (w[0]-c[0])*q0+(w[1]-c[1])*q1      
      
def empirical_profit(s, c, h_demand, w):
    h_demand=h_demand.copy()
    h_demand0=[]
    h_demand1=[]
    for i in range(len(h_demand)):
        h_demand0.append(h_demand[i][0])
        h_demand1.append(h_demand[i][1])
    h_demand0.sort() #对需求升序排列
    h_demand1.sort() #对需求升序排列
    location0=math.ceil(len(h_demand0)*(1-w[0]/s[0])) #确定需要累积的需求个数
    q0 = h_demand0[location0-1] #根据FOC计算q
    location1=math.ceil(len(h_demand1)*(1-w[1]/s[1])) #确定需要累积的需求个数
    q1 = h_demand1[location1-1] #根据FOC计算q
    # print("经验分布收益", q0,q1, (w[0]-c[0])*q0+(w[1]-c[1])*q1)
    return (w[0]-c[0])*q0+(w[1]-c[1])*q1 #返回收益

def center_profit(s, c, xi, eta, w):
    xi=xi.copy()
    eta=eta.copy()
    d=zip(xi, eta)
    d=list(d)
    d.sort(key=lambda x: x[0][0], reverse=False)  # 按照元组的第一个数排
    xi0, eta0 = zip(*d)
    xi0=list(xi0)
    eta0=list(eta0)
    d.sort(key=lambda x: x[0][1], reverse=False)  # 按照元组的第2个数排
    xi1, eta1 = zip(*d)
    i0=1
    while i0 <= len(xi0):
        if sum(eta0[:i0])>=1-w[0]/s[0]:
            xi_w0=xi0[i0-1][0]
            break
        i0=i0+1#计算xi_w
    i1=1
    while i1 <= len(xi1):
        if sum(eta1[:i1])>=1-w[1]/s[1]:
            xi_w1=xi1[i1-1][1]
            break
        i1=i1+1#计算xi_w
    return xi_w0, xi_w1,(w[0]-c[0])*xi_w0+(w[1]-c[1])*xi_w1 #返回收益

def order_worstprofit(c, h_order, h_w, w):
    h_order=h_order.copy()
    h_w=h_w.copy()  
    h_order0=[]
    h_order1=[]
    h_w0=[]
    h_w1=[]
    for i in range(len(h_w)):
        h_order0.append(h_order[i][0])
        h_order1.append(h_order[i][1])
        h_w0.append(h_w[i][0])
        h_w1.append(h_w[i][1])
    h_w0.sort()#对历史订单排序
    h_w1.sort()#对历史订单排序
    h_order0=sorted(h_order0,reverse=True)
    h_order1=sorted(h_order1,reverse=True)
    lw0 = 0
    lw_index0=0
    for i in h_w0:
        if i > w[0]:
            lw0=i
            q0=h_order0[lw_index0]
            break
        lw_index0=lw_index0+1
    if lw0==0:
        q0=0      
    lw1 = 0
    lw_index1=0
    for j in h_w1:
        if j > w[1]:
            lw1=j
            q1=h_order1[lw_index1]
            break
        lw_index1=lw_index1+1
    if lw1==0:
        q1=0    
    return (w[0]-c[0])*q0+(w[1]-c[1])*q1

def discrete_support_q(edge,q_low,q_up):####将需求空间离散化
    if q_up[0]%edge!=0:
        q_up[0] = (q_up[0]//edge+1)*edge
    li0 = range(q_low[0], q_up[0]+1, edge)
    if q_up[1]%edge!=0:
        q_up[1] = (q_up[1]//edge+1)*edge
    li1 = range(q_low[1], q_up[1]+1, edge)
    li1 = range(q_low[1], q_up[1]+1, edge)
    discrete_support=list(itertools.product(li0,li1,repeat=1))
    return discrete_support

def feasibility(s, xi, q, w, eta):##LP model single
    xi=xi.copy()
    xi0=[]
    xi1=[]
    for i in range(len(xi)):
        xi0.append(xi[i][0])
        xi1.append(xi[i][1])
    epsilon = 0.00000001
    support0=[]
    support1=[]
    support0=xi0+[q[0]]
    support1=xi1+[q[1]]
    support=[]
    for t in range(len(support0)):
        for j in range(len(support1)):
            if support0[t] not in xi0 or support1[j] not in xi1:
                support.append([support0[t],support1[j]])
    for i in range(len(xi)):
        support.append(xi[i])
    q0_index=[]
    q1_index=[]
    q00_index=[]
    q11_index=[]
    index0=0
    for a in range(len(support)):
        if support[a][0]<=q[0]:
            q0_index.append(index0)
        index0=index0+1
    index0=0
    for a in range(len(support)):
        if support[a][0] < q[0]:
            q00_index.append(index0)
        index0=index0+1
    index1=0
    for e in range(len(support)):
        if support[e][1]<=q[1]:
            q1_index.append(index1)
        index1=index1+1
    index1=0
    for e in range(len(support)):
        if support[e][1]<q[1]:
            q11_index.append(index1)
        index1=index1+1
    K = range(len(support))
    time=range(len(xi))
    m = Model()
    P = m.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="p")
    pai = m.addVars(time, K, lb=0, vtype=GRB.CONTINUOUS, name="pai")#辅助变量，用于处理目标函数
    m.setObjective(sum(((xi[t][0]-support[k][0])**2+(xi[t][1]-support[k][1])**2)\
                       *pai[t,k] for k in K for t in time), GRB.MINIMIZE)
    m.addConstrs((sum(pai[t, k] for k in K) == eta[t] for t in time), name="r")  # 9a
    m.addConstrs((sum(pai[t, k] for t in time) == P[k] for k in K), name="r1")  # 9b
    m.addConstr((sum(P[k] for k in q0_index) >= 1 - w[0] / s[0]), name='c')#9c
    m.addConstr((sum(P[k] for k in q00_index) <= 1 - w[0] / s[0] - epsilon), name='c')#9d
    m.addConstr((sum(P[k] for k in q1_index) >= 1 - w[1] / s[1]), name='c')#9c
    m.addConstr((sum(P[k] for k in q11_index) <= 1 - w[1] / s[1] - epsilon), name='c')#9d
    m.Params.LogToConsole = 0
    m.optimize()
    # print('--------modelStatus', m.Status)
    if m.Status != 2:
        return [], [] #模型无解则返回空值
    return m.objVal

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

def cutting_plane(edge, c, h_demand, h_order, h_w, w, s, order_up, order_low, theta):
    h_demand=h_demand.copy()
    h_order=h_order.copy()
    h_w=h_w.copy()  
    h_order0=[]
    h_order1=[]
    h_w0=[]
    h_w1=[]
    for i in range(len(h_w)):
        h_order0.append(h_order[i][0])
        h_order1.append(h_order[i][1])
        h_w0.append(h_w[i][0])
        h_w1.append(h_w[i][1])
    h_w0.append(c[0])
    h_w0.append(s[0])
    h_w1.append(c[1])
    h_w1.append(s[1])
    h_order0.append(order_low[0])
    h_order0.append(order_up[0])
    h_order1.append(order_low[1])
    h_order1.append(order_up[1])
    h_w0.sort()#对历史订单排序
    h_w1.sort()#对历史订单排序
    h_order0=sorted(h_order0,reverse=True)
    h_order1=sorted(h_order1,reverse=True)
    lw_index0=0
    for i in h_w0:
        if i > w[0]:
            q0=h_order0[lw_index0]
            q0_=h_order0[lw_index0-1]
            break
        lw_index0=lw_index0+1 
    lw_index1=0
    for j in h_w1:
        if j > w[1]:
            q1=h_order1[lw_index1]
            q1_=h_order1[lw_index1-1]
            break
        lw_index1=lw_index1+1
    uuu=[math.ceil(q0_),math.ceil(q1_)]
    lll=[int(q0),int(q1)]
    O_hat=discrete_support_q(edge,lll,uuu)
    xi,eta=get_center(s, h_demand, h_order, h_w)
    q_cen0,q_cen1,profit_cen=center_profit(s, c, xi, eta, w)
    q_c=(q_cen0,q_cen1)
    q=(q_cen0,q_cen1)
    O_hat.append(q_c)
    O_H=[]
    for v in O_hat:
        if sum((w[i]-c[i])*v[i] for i in range(2)) <= sum((w[i]-c[i])*q_c[i] for i in range(2)):
            O_H.append(v)
    O_hat=O_H###删除比球心对应的q收益更高的q  
    UPBOUND=sum((w[i]-c[i])*q_c[i] for i in range(2))##设置上界
    lb=[]
    for l in range(len(O_hat)):
        lb.append(sum((w[i]-c[i])*O_hat[l][i] for i in range(2)))
    lb.sort()
    LOWERBOUND=lb[0]##设置下界
    tolerance=0.01
    count = 0
    while (UPBOUND - LOWERBOUND > tolerance) and (len(O_hat)>1):
        count += 1
        OO=[]
        O1=[]
        num_items = len(O_hat)
        random_index = random.randrange(num_items)
        q_i = O_hat[random_index]
        fea=feasibility(s, xi, q_i, w, eta)
        if fea<=theta*theta:###模型20可行
            if sum((w[i]-c[i])*q_i[i] for i in range(2)) <= UPBOUND:
                UPBOUND=sum((w[i]-c[i])*q_i[i] for i in range(2))
                q=q_i
                for j in O_hat:
                    if sum((w[i]-c[i])*j[i] for i in range(2)) <= UPBOUND:
                        OO.append(j)###删除收益更高的q 
                O_hat=OO
        else:
            for k in O_hat:
                if k[0]>q_i[0] or k[1]>q_i[1]:
                    O1.append(k)
            O_hat=O1#############根据第一个产品不可行排除
        if len(O_hat)<=1:
            break
        Lb=[]
        for r in range(len(O_hat)):
            Lb.append(sum((w[i]-c[i])*O_hat[r][i] for i in range(2)))
        Lb.sort()
        LOWERBOUND=Lb[0]###算法3step22，更新下界
    return q, UPBOUND

def demand_cutting_plane(edge, c, h_demand, w, s, order_up, order_low, theta):
    h_demand=h_demand.copy()
    uuu=[math.ceil(order_up[0]),math.ceil(order_up[1])]
    lll=[int(order_low[0]),int(order_low[1])]
    O_hat=discrete_support_q(edge,lll,uuu)
    xi=h_demand.copy()
    eee=[1/len(h_demand) for i in range (len(h_demand))]
    eta=eee.copy()
    q_cen0,q_cen1,profit_cen=center_profit(s, c, xi, eta, w)
    q_c=(q_cen0,q_cen1)
    q=(q_cen0,q_cen1)
    O_hat.append(q_c)
    O_H=[]
    for v in O_hat:
        if sum((w[i]-c[i])*v[i] for i in range(2)) <= sum((w[i]-c[i])*q_c[i] for i in range(2)):
            O_H.append(v)
    O_hat=O_H###删除比球心对应的q收益更高的q  
    UPBOUND=sum((w[i]-c[i])*q_c[i] for i in range(2))##设置上界
    lb=[]
    for l in range(len(O_hat)):
        lb.append(sum((w[i]-c[i])*O_hat[l][i] for i in range(2)))
    lb.sort()
    LOWERBOUND=lb[0]##设置下界
    tolerance=0.01
    count = 0
    while (UPBOUND - LOWERBOUND > tolerance) and (len(O_hat)>1):
        count += 1
        OO=[]
        O1=[]
        num_items = len(O_hat)
        random_index = random.randrange(num_items)
        q_i = O_hat[random_index]
        fea=feasibility(s, xi, q_i, w, eta)
        if fea<=theta*theta:###模型20可行
            if sum((w[i]-c[i])*q_i[i] for i in range(2)) <= UPBOUND:
                UPBOUND=sum((w[i]-c[i])*q_i[i] for i in range(2))
                q=q_i
                for j in O_hat:
                    if sum((w[i]-c[i])*j[i] for i in range(2)) <= UPBOUND:
                        OO.append(j)###删除收益更高的q 
                O_hat=OO
        else:
            for k in O_hat:
                if k[0]>q_i[0] or k[1]>q_i[1]:
                    O1.append(k)
            O_hat=O1#############根据第一个产品不可行排除
        if len(O_hat)<=1:
            break
        Lb=[]
        for r in range(len(O_hat)):
            Lb.append(sum((w[i]-c[i])*O_hat[r][i] for i in range(2)))
        Lb.sort()
        LOWERBOUND=Lb[0]###算法3step22，更新下界
    return q, UPBOUND

def experiment(s, c, mu, sigma, correlation, edge, demand_low, demand_up, n, T_d, T_o, theta):
    l = 0
    W=generate_W(s, c)    # w用来控制枚举的批发价格
    df = pd.DataFrame()
    for r in range(n):
        H_demand,H_order,H_w=get_H_data(mu, sigma, correlation, s, c, 5, 20, demand_low, demand_up)
        h_demand=[]
        h_order=[]
        h_w=[]
        for d in range(T_d):
            h_demand.append(H_demand[d])
        for o in range(T_o):
            h_order.append(H_order[o])
            h_w.append(H_w[o])
        print('历史数据',h_demand,h_order,h_w)
        xi, eta = get_center(s, h_demand, h_order, h_w)
        print('球心分布',xi,eta)
        to_2 = lambda x: str(list(map(lambda a: round(a, 2), x)))
        h1=to_2(h_demand[0])
        h2=to_2(h_order[0])
        h3=to_2(h_w[0])
        for i in range(1,T_o):
            h2=h2+to_2(h_order[i])
            h3=h3+to_2(h_w[i])
        for i in range(1,T_d):
            h1=h1+to_2(h_demand[i])
        empirical_w=[0,0]
        center_w=[0,0]
        order_w=[0,0]
        demand_w=[0,0]
        both_w=[0,0]
        empirical_p=0
        center_p=0
        order_p=0
        demand_p=0
        both_p=0
        for w in W:
            print('wwwwwwwwwwwwwwwwwwww',w)
            empirical_prof=empirical_profit(s, c, h_demand, w)
            if empirical_prof > empirical_p:
                empirical_p=empirical_prof
                empirical_w[0]=w[0]
                empirical_w[1]=w[1]
            a,b,center_prof=center_profit(s, c, xi, eta, w)
            if center_prof > center_p:
                center_p=center_prof
                center_w[0]=w[0]
                center_w[1]=w[1]
            order_prof=order_worstprofit(c, h_order, h_w, w)
            if order_prof>order_p:
                order_p=order_prof
                order_w[0]=w[0]
                order_w[1]=w[1]
            e,demand_prof=demand_cutting_plane(edge, c, h_demand, w, s, demand_up, demand_low, theta)
            if demand_prof>demand_p:
                demand_p=demand_prof
                demand_w[0]=w[0]  
                demand_w[1]=w[1]      
            q,both_prof=cutting_plane(edge, c, h_demand, h_order, h_w, w, s, demand_up, demand_low, theta)
            if both_prof>both_p:
                both_p=both_prof
                both_w[0]=w[0]
                both_w[1]=w[1]
        true_profit_em=true_profit(mu, sigma, correlation, empirical_w, s, c)
        true_profit_ce=true_profit(mu, sigma, correlation, center_w, s, c)
        true_profit_or=true_profit(mu, sigma, correlation, order_w, s, c)
        true_profit_de=true_profit(mu, sigma, correlation, demand_w, s, c)
        true_profit_bo=true_profit(mu, sigma, correlation, both_w, s, c)
        data = dict()
        data['T_d'] = T_d
        data['T_o'] = T_o
        data['demand'] = h1
        data['order'] = h2
        data['w'] = h3
        data['empirical_w'] = empirical_w
        data['true_profit_em'] = true_profit_em
        data['center_w'] = center_w
        data['true_profit_ce'] = true_profit_ce
        data['order_w'] = order_w
        data['true_profit_or'] = true_profit_or
        data['demand_w'] = demand_w
        data['true_profit_de'] = true_profit_de
        data['both_w'] = both_w
        data['true_profit_bo'] = true_profit_bo
        df[l] = pd.Series(data)
        l+=1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("试验次数",l)
    em_avg=mean(list(df.loc['true_profit_em']))
    em_var=np.var(list(df.loc['true_profit_em']))
    ce_avg=mean(list(df.loc['true_profit_ce']))
    ce_var=np.var(list(df.loc['true_profit_ce']))
    or_avg=mean(list(df.loc['true_profit_or']))
    or_var=np.var(list(df.loc['true_profit_or']))
    de_avg=mean(list(df.loc['true_profit_de']))
    de_var=np.var(list(df.loc['true_profit_de']))
    bo_avg=mean(list(df.loc['true_profit_bo']))
    bo_var=np.var(list(df.loc['true_profit_bo']))
    df.at['true_profit_em',n+1]=em_avg
    df.at['true_profit_em',n+2]=em_var
    df.at['true_profit_ce',n+1]=ce_avg
    df.at['true_profit_ce',n+2]=ce_var
    df.at['true_profit_or',n+1]=or_avg
    df.at['true_profit_or',n+2]=or_var
    df.at['true_profit_de',n+1]=de_avg
    df.at['true_profit_de',n+2]=de_var
    df.at['true_profit_bo',n+1]=bo_avg
    df.at['true_profit_bo',n+2]=bo_var
    print(df)
    df.to_csv('相关-订单+鲁棒d5-o20.csv')

start_time=time.time()  
demand_low=[0,0]
demand_up=[100,100]
mu=[50,50]
sigma=[30,20]
T_d=5
T_o=20
c=[10,10]
s=[40,30]
theta=2
n=20
experiment(s, c, mu, sigma, 0.3, 2, demand_low, demand_up, n, T_d, T_o, theta)
end_time=time.time() 
print('总时间',end_time-start_time)



