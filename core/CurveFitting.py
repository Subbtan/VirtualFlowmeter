import numpy as np
import scipy as sc
import scipy.optimize as opt
import csv
import matplotlib.pyplot as plt

# 导入数据

fpath = "D:\MyProject\VritualFlowmeter\sources\PumpPerformanceCurve32-80.csv"

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

def residuals2(p):
    a1,a2,a3 = p
    r = Y - (a1*X**2 + a2*X**1 + a3)
    return r

def residuals3(p):
    a1,a2,a3,a4 = p
    r = Y - (a1*X**3 + a2*X**2 + a3*X + a4)
    return r

# 最小二乘求系数

r = opt.leastsq(residuals2,[1,0,1])

a1,a2,a3 = r[0]

Yfit = a1*X**2 + a2*X**1 + a3

ccpfh = sum(residuals2(r[0])**2) #残差平方和

def draw2():
    plt.plot(data_Q,data_H,'r')
    plt.plot(data_Q,Yfit,'b')
    pass
