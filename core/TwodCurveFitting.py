# 拟合过程
# 常规import
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import csv

from Package import FittingMethod as Ft 

# 特殊import
from scipy.linalg import lstsq #好用的最小二乘方法



#导入数据

fpath = "../sources/PPCall.csv"
with open(fpath,'r',encoding='utf-8') as f:
    f_csv = csv.reader(f)
    rawData = []
    for row in f_csv:
        rawData.append(row)
    rawData=rawData[1:]

# 数字化
# 按列分
x = []
z = []
for row in rawData:
    row[0]=int(row[0])
    row[1]=float(row[1])
    row[2]=float(row[2])
    x.append(row[0:2])
    z.append(row[2])
x=np.array(x)
z=np.array(z)

'''
# 构造幂
def Degrees(n):
    ls=[] #创建数列存放幂对儿
    for k in range(n+1):
        for i in range(k+1):
            ls.append((i,k-i))
    return ls[::-1]

# 构造A矩阵
def MkMatrix(x,deg):
    A = np.stack([np.prod(x**d,axis=1) for d in deg],axis=-1)
    return A

# 计算结果
def CalAnswer(x,deg,coe): #输入
    A = MkMatrix(x,deg)
    ans = np.sum(coe * A , axis=1)
    return ans
'''
'''# 通过import自己写的module代替'''

deg = Ft.Degrees(3)
A = Ft.MkMatrix(x,deg)

# 进行拟合
def run():
    deg = Ft.Degrees(3)
    A = Ft.MkMatrix(x,deg)
    c, resid, rank, sigma = Ft.lstsq(A, z)
    return [c,resid,rank,sigma]

# 画图看看与结果分析

from mpl_toolkits.mplot3d import Axes3D
