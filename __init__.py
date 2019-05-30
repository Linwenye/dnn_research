import matplotlib.pyplot as plt
import numpy
import math

# 创建x数组，区间是[0,4π]，步长为0.1
x = numpy.arange(0, 4 * math.pi, 0.1)


def sinFunction(x):
    sin = [math.sin(i) for i in x]
    return sin


def cosFunction(x):
    cos = [math.cos(i) for i in x]
    return cos


plt.scatter(x, sinFunction(x), label='sin')
plt.plot(x, cosFunction(x), label='cos')
plt.legend()
plt.show()
