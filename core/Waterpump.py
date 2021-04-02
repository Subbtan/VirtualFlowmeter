import numpy as np
import scipy as sc
import matplotlib as mpl


class Waterpump:

    def __init__(self, name, coe_H, coe_N, coe_eta):
        # coe_H coe_N 为
        # 定义水泵 包含水泵名称、三个特性曲面 十参数
        # H/N/eta=f(Q,n)

        self.coe_eta = np.zeros(10)
        self.coe_H = np.zeros(10)
        self.coe_N = np.zeros(10)

        self.name = name
        self.coe_H = coe_H
        self.coe_N = coe_N
        self.coe_eta = coe_eta

        pass
'''
    def GetData(self, Q, n, Type):
       # ans = f(Q, n)


        # 根据输入流量与转速，返回H/N/eta的数值

        return ans
'''

print('1')

def prt():
    print("hello world!")

    pass
