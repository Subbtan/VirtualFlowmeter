# 试试怎么画三维图
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import csv

# from ..core import TwodCurveFitting

# 画图
from mpl_toolkits.mplot3d import Axes3D

# mpl.rcParams['legend.fontsize'] = 10
'''
def PrintCurve(n=xx0,Q=xx1,H=zz):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(n,Q,H, label='parametric curve')
    ax.legend()
    plt.show()
'''
# PrintCurve()

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X,Y,Z)








