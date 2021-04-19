import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from scipy.optimize.minpack import leastsq #(最小二乘，只能用单变量)
from scipy.linalg import lstsq #好用的最小二乘方法

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

# 进行拟合
#c, resid, rank, sigma = lstsq(A, z)


# 计算结果
def CalAnswer(x,deg,coe): #输入
    A = MkMatrix(x,deg)
    ans = np.sum(coe * A , axis=1)
    return ans

def error(x,z,deg,coe):
    error = z-CalAnswer(x,deg,coe)
    return error
    


def CalforGrid(X,Y,deg,coe):
    ans = 0
    if len(deg) == len(coe):
        for i in range(len(deg)):
            temp = coe[i]*((X**deg[i][0])*(Y**deg[i][1]))
            ans += temp

        return ans


