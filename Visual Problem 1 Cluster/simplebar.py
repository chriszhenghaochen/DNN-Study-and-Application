import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import sys

class barchart(object):

    def __init__(self, names1 = None,  filename = 'freqs_Fri.csv', labourname = 'E:/PycharmProjects/python-visualising-jeremy/datain/key points.csv'):
        self.names1 = names1
        # self.names2 = names2
        self.file = pd.read_csv(filename)
        self.labour = pd.read_csv(labourname)
        print 'barchart set up'

    def groupmeanstd(self, names):

        result = []
        rows = []
        sum = []

        for i in range(len(names)):
            row = self.file[self.file.id == names[i]].values
            row = row.ravel()[1:72]
            rows.append(row)

        rows = np.array(rows)

        #1 thrill
        thrillrows1 = rows[:,:8]
        thrillrows2 = rows[:,66:67]

        thrillrows = np.concatenate((thrillrows1, thrillrows2), axis=1)

        sum.append(np.sum(thrillrows, axis= 1))

        #2 kidle
        KiddieRides = rows[:,8:19]

        sum.append(np.sum( KiddieRides, axis= 1))

        #3 RidesforEveryone
        RidesforEveryone = rows[:,19:31]

        sum.append(np.sum(RidesforEveryone, axis= 1))

        #4 ShowsEntertainment
        ShowsEntertainment1 = rows[:,31:32]
        ShowsEntertainment2 = rows[:,59:60]
        ShowsEntertainment3 = rows[:,61:63]

        ShowsEntertainment = np.concatenate((ShowsEntertainment1, ShowsEntertainment2), axis=1)
        ShowsEntertainment = np.concatenate((ShowsEntertainment, ShowsEntertainment3), axis=1)

        sum.append(np.sum(ShowsEntertainment, axis= 1))

        #5 BeerGardens
        BeerGardens = rows[:,32:34]
        sum.append(np.sum(BeerGardens, axis= 1))

        #6 food
        food1 = rows[:,34:39]
        food2 = rows[:,52:59]

        food = np.concatenate((food1, food2), axis=1)

        sum.append(np.sum(food, axis= 1))

        #7 shopping
        shopping = rows[:,39:48]

        sum.append(np.sum(shopping, axis= 1))

        #8 RestRoom
        restroom1 = rows[:,48:52]
        restroom2 = rows[:,63:66]

        restroom = np.concatenate((restroom1,restroom2),axis=1)

        sum.append(np.sum(restroom, axis=1))

        #9 Information & Assistance
        InformationAssistance = rows[:,60:61]

        sum.append(np.sum(InformationAssistance, axis=1))

        #10 Outdoor Area
        outdoorArea = rows[:,67:70]

        sum.append(np.sum(outdoorArea, axis=1))

        #11 Entry/Exit

        entryexit = rows[:,70:73]

        sum.append(np.sum(entryexit, axis=1))
        sum = np.array(sum)

        result.append(sum.mean(axis = 1))
        result.append(sum.std(axis = 1))

        result = np.array(result)

        return result


    def meanstd(self, names):

            result = []
            rows = []

            for i in range(len(names)):
                row = self.file[self.file.id == names[i]].values
                row = row.ravel()[1:72]
                rows.append(row)

            rows = np.array(rows)

            result.append(rows.mean(axis=0))
            result.append(rows.std(axis = 0))

            result = np.array(result)

            return result


    def process(self, grouping = 'No'):

        if grouping == 'No':
            result1 = self.meanstd(self.names1)
            # result2 = self.meanstd(self.names2)
            N = 70

        else:
            result1 = self.groupmeanstd(self.names1)
            # result2 = self.groupmeanstd(self.names2)
            category = self.labour['category']
            N = 11

        menMeans = result1[0]
        menStd = result1[1]

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

        # womenMeans = result2[0]
        # womenStd = result2[1]
        # rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Freqence')
        ax.set_title('Frequence of Friday')

        if grouping == 'No':
            ax.set_xticks(ind + width)
            ax.set_xticklabels(self.labour['name'].values)

        else:
            ax.set_xticks(ind + width)
            ax.set_xticklabels(np.unique(category))

        groupname1 = ''
        # groupname2 = ''

        for i in range(len(self.names1)):
            groupname1 += str(self.names1[i])
            groupname1 += ' '

        # for i in range(len(self.names2)):
        #     groupname2 += str(self.names2[i])
        #     groupname2 += ' '


        # red_patch = mpatches.Patch(color='red', label= groupname1)
        #
        # plt.legend(handles=[red_patch], prop = {'size':10})
        # ax.legend((rects1[0], rects1[0]), (groupname1, groupname1))
        # ax.legend((rects1[0]), (groupname1))

        plt.show()