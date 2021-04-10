import numpy as np
import scipy as sc
import scipy.optimize as opt
import csv
import matplotlib.pyplot as plt
from scipy.optimize.minpack import leastsq

# 导入数据

# fpath = "D:\MyProject\VritualFlowmeter\sources\PumpPerformanceCurve32-80.csv"
fpath = "../sources/PumpPerformanceCurve32-80.csv"
with open(fpath,'r') as f:
    f_csv=csv.reader(f)
    
    num = []
    Q = []
    H = []
    X=[]
    
    for row in f_csv:
        # print(row)
        # print(type(row))

        num.append(row[0])
        Q.append(row[1])
        H.append(row[2])
    #数字化
    Q=Q[1:-2];H=H[1:-2]
    Q=[float(i) for i in Q]
    H=[float(i) for i in H]


data_Q = np.array(Q) 
data_H = np.array(H)

# 画图看看

def draw1():
    plt.plot(data_Q,data_H)
    plt.show()
    pass


# 最小二乘法拟合

# 误差函数
'''
def residuals(p,X=data_Q,Y=data_H):
    if len(p) == 3:
        a1,a2,a3 = p
        r = Y - (a1*X^2 + a2*X^1 + a3)
        return r

    elif len(p) == 4:
        a1,a2,a3,a4 = p
        r = Y - (a1*X^3 + a2*X^2 + a3*X + a4)
        return r

    else : print("参数错误")
'''

X = data_Q
Y = data_H

#
def PolyResiduals(p,x,y):
    fun = np.poly1d(p)
    return y - fun(x)

def PowerResiduals(p,x,y):
    a,b,c = p
    return y - a*x**b + c


def fitting(p):
    if p >= 1:
        pars = np.random.rand(p+1)
        r = leastsq(PolyResiduals,pars,args=(X,Y))
        return r[0]  #返回系数
    
    elif p == -1:
        pars = np.random.rand(3)
        pars = [-1.488e-07,4.983,11]
        r = leastsq(PowerResiduals,pars,args=(X,Y))
        return r[0]

# 结果分析
def result(p):
    if p >= 1:
        coe = fitting(p)
        fun = np.poly1d(coe)

        plt.plot(data_Q,data_H,'r')
        plt.plot(data_Q,fun(data_Q),'b')
        plt.show()

    if p == -1:
        coe = fitting(p)
        a,b,c = coe
        y = a*data_Q**b + c
        plt.plot(data_Q,data_H,'r')
        plt.plot(data_Q,y,'b')
        plt.show()

# Yfit = a1*X**2 + a2*X**1 + a3

# ccpfh = sum(residuals(r[0])**2) #残差平方和


# 拟合优度计算



class CurveFitting:
#  本对象目的：进行拟合，返回拟合结果（系数），输出拟合图像，计算拟合优度

    def __init__(self,data_Q,data_N,data_H):
        self.X = data_Q
        self.Y = data_N
        self.Z = data_H
        pass

    def residuals(self,p,x,y):
        fun = np.poly1d(p)
        return y - fun(x)

    def fitting(self,p):
        pars = np.random.rand(p+1)
        r = leastsq(PolyResiduals, pars, args=(X, Y))
        return r[0]  # 返回系数

