import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import math

fig, ax = plt.subplots()

img = mpimg.imread('C:/Users/zchen4/Desktop/data/Park Map.jpg')
imgplot = ax.imshow(img, extent=[0,100,0,100])


with open('C:/Users/zchen4/Desktop/data/id 941.csv') as f1:
    data1 = f1.read()

with open('C:/Users/zchen4/Desktop/data/trace0.csv') as f2:
    data2 = f2.read()


data1 = data1.split('\n')
x1 = [row.split(',')[0] for row in data1]
y1 = [row.split(',')[1] for row in data1]


data2 = data2.split('\n')
x2 = [row for row in data2]

# print x2

l1 = len(x1)
l2 = len(x2)

if l1 != l2:
    print "no equal"

# data = np.array([x, y], np.int32)


# print data
#
# def animate(i,j):
#
def animate(i):
    if i < l1 and i > 1:
        x1_sub = x1[i-2:i]
        y1_sub = y1[i-2:i]
        speed = math.pow(float(x2[i]),2)

        # #thickness
        # ax.plot(x1_sub, y1_sub, c='black', linewidth = speed * 100)

        # #color
        ax.plot(x1_sub, y1_sub, c = (1 - speed*10, 1-speed*10,1 -speed*10), linewidth = 2)

    return ax


def init():
    return ax


ani = animation.FuncAnimation(fig, animate, init_func=init, blit=False,
   interval=10, repeat=True)

plt.show()