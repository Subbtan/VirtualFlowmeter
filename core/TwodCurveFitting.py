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
# 得到完整拟合结果
def Fitting(x,z,n=3):
    deg = Ft.Degrees(n)
    A = Ft.MkMatrix(x,deg)
    c, resid, rank, sigma = Ft.lstsq(A, z)
    return [c,resid,rank,sigma]

# 单纯获得系数
def getCoe(x,z,n=3):
    deg = Ft.Degrees(n)
    A = Ft.MkMatrix(x,deg)
    c, resid, rank, sigma = Ft.lstsq(A, z)
    return c

coe = getCoe(x,z)

# 画图看看与结果分析
# x,y 代表原始数据 X，Y，Z 大写代表网格数据（手动生产的）

xx0 = np.linspace(60,610,200)
xx1 = np.linspace(0,120,200)
X,Y = np.meshgrid(xx0,xx1)
Z = Ft.CalforGrid(X,Y,deg,coe)

# 误差分析
# Ft.error(x,z,deg,coe)

# 测试各阶误差有多少   返回拟合优度
def test(x,z,n):
    deg = Ft.Degrees(n)
    coe = getCoe(x,z,n)
    er = Ft.error(x,z,deg,coe)
    R2 = sum(er**2)   # 残差平方和
    percentage = abs(er/Ft.CalAnswer(x,deg,coe))
    avgper = sum(percentage)/len(percentage) * 100
    maxper = max(percentage) * 100
    return [R2,avgper,maxper,coe]

test(x,z,6)

for i in range(10):
    i+=1
    temp = test(x,z,i)
    print(i,temp[:3])