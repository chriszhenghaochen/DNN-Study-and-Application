import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import colorsys
import sys

class HeatMap(object):
    def __init__(self, names, names2 = [],  filename = 'freqs_Fri.csv', labourname = 'E:/PycharmProjects/python-visualising-jeremy/datain/key points.csv', mapname = 'E:/PycharmProjects/python-visualising-jeremy/datain/image/Park Map - Light.jpg'):
        self.names = names
        self.names2 = names2
        self.img = mpimg.imread(mapname)
        self.file = pd.read_csv(filename)
        self.points = pd.read_csv(labourname).ix[1:75,3:5].values

        print "heat map set up"


    def normilise(self, array):

        maxval = max(array)
        minval = min(array)
        result = []

        for i in range(len(array)):
            result.append(float((array[i] - minval))/float((maxval - minval)))

        return result

    def process(self):

        plt.imshow(self.img, extent=[0,100,0,100])
        rows = []
        colors = []

        #values, i.e. means of values
        for i in range(len(self.names)):
            row = self.file[self.file.id == self.names[i]].values
            row = row.ravel()[1:72]
            rows.append(row)

        rows = np.array(rows)
        sum1 = rows.sum(axis=0)
        # print sum1

        sum2 = self.normilise(sum1)
        # print sum2

        for i in range(len(sum2)):
            colors.append(colorsys.hsv_to_rgb(sum2[i],  sum2[i], sum2[i]))

        plt.scatter(self.points[:,0], self.points[:,1], s= 100*sum1, color= colors)
        # plt.scatter(self.points[:,0], self.points[:,1], s= sum1/10, color= colors)

        #legend
        groupname1 = ''

        for i in range(len(self.names)):
            groupname1 += str(self.names[i])
            groupname1 += ' '


        red_patch = mpatches.Patch(color='red', label= groupname1)
        plt.legend(handles=[red_patch], prop = {'size':10})
        plt.show()

    def groupProcess(self):
        fig, axes = plt.subplots(ncols= 2)
        axes[0].imshow(self.img, extent=[0,100,0,100])
        axes[1].imshow(self.img, extent=[0,100,0,100])


        #names1  values, i.e. means of values
        rows = []
        colors = []
        for i in range(len(self.names)):
            row = self.file[self.file.id == self.names[i]].values
            row = row.ravel()[1:72]
            rows.append(row)

        rows = np.array(rows)
        sum1 = rows.sum(axis=0)
        # print sum1

        sum2 = self.normilise(sum1)
        # print sum2

        for i in range(len(sum2)):
            colors.append(colorsys.hsv_to_rgb(sum2[i],  sum2[i], sum2[i]))

        axes[0].scatter(self.points[:,0], self.points[:,1], s= 100*sum1, color= colors)

        # axes[0].scatter(self.points[:,0], self.points[:,1], s= sum1/10, color= colors)

        #legend
        groupname1 = ''

        for i in range(len(self.names)):
            groupname1 += str(self.names[i])
            groupname1 += ' '


        # red_patch = mpatches.Patch(color='red', label= groupname1)
        # axes[0].legend(handles=[red_patch], prop = {'size':10})

        #nemas2 values, i.e. means of values
        rows = []
        colors = []

        for i in range(len(self.names2)):
            row = self.file[self.file.id == self.names2[i]].values
            row = row.ravel()[1:72]
            rows.append(row)

        rows = np.array(rows)
        sum1 = rows.sum(axis=0)
        # print sum1

        sum2 = self.normilise(sum1)
        # print sum2

        for i in range(len(sum2)):
            colors.append(colorsys.hsv_to_rgb(sum2[i],  sum2[i], sum2[i]))

        axes[1].scatter(self.points[:,0], self.points[:,1], s= 100*sum1, color= colors)
        # axes[1].scatter(self.points[:,0], self.points[:,1], s= sum1/10, color= colors)

        #legend
        groupname1 = ''

        for i in range(len(self.names2)):
            groupname1 += str(self.names2[i])
            groupname1 += ' '


        # red_patch = mpatches.Patch(color='red', label= groupname1)
        # axes[1].legend(handles=[red_patch], prop = {'size':10})
        plt.show()

