#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("E:/data/dtwoutput.txt") as f:
    data = f.read()

with open("E:/data/dtwdistance.txt") as f2:
    data2 = f2.read()

data = data.split('\n')
data = data[:len(data) - 1]
distance = data2

x = [row.split(',')[0] for row in data]
y = [row.split(',')[1] for row in data]

label = 'distance = ' + distance

red_patch = mpatches.Patch(color='red', label= label)
plt.legend(handles=[red_patch], prop = {'size':10})

plt.plot(x,y, c='r', label='the data')
plt.show()