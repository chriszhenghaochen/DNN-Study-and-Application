import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import dateutil


class communication(object):

     def __init__(self, filename = 'comm-data-Fri.csv', group1 = [], group2 = [],grouping = 'No'):
        self.group1 = group1
        self.group2 = group2
        self.grouping = grouping

        file = pd.read_csv(filename)

        #call from 1 to 2
        self.catch1 = file[file['from'].isin(map(int,self.group1)) & file['to'].isin(map(str,self.group2))]

        #call from 2 to 1
        self.catch2 = file[file['from'].isin(map(int,self.group2)) & file['to'].isin(map(str,self.group1))]

        if self.grouping == 'Yes':
            #call from 1 to 1
            self.catch3 = file[file['from'].isin(map(int,self.group1)) & file['to'].isin(map(str,self.group1))]

            #call from 1 to 1
            self.catch4 = file[file['from'].isin(map(int,self.group2)) & file['to'].isin(map(str,self.group2))]

        print "communication chart set up"

     def singleprocess(self, catch1, catch2, axes):

         com1 = catch1.groupby('Timestamp').agg(['count'])
         com2 = catch2.groupby('Timestamp').agg(['count'])

         val1 = com1['from'].values.ravel()
         time1 = com1.index.values.ravel()
         date1 = [dateutil.parser.parse(s) for s in time1]


         val2 = com2['from'].values.ravel()
         time2 = com2.index.values.ravel()
         date2 = [dateutil.parser.parse(s) for s in time2]

         time = np.concatenate((time1, time2), axis=0)
         date = [dateutil.parser.parse(s) for s in time]

         ax = plt.gca()
         ax.set_xticks(date)
         xfmt = md.DateFormatter('%H:%M:%S')
         ax.xaxis.set_major_formatter(xfmt)

         axes.plot(date1, val1, c = 'blue')
         axes.plot(date2, val2, c = 'black')
         return axes


     def process(self):

        if self.grouping == 'No':
            self.singleprocess(self.catch1, self.catch2, plt).show()

        else:
            fig, axes = plt.subplots(nrows= 2)
            self.singleprocess(self.catch3, self.catch3,axes[0])
            self.singleprocess(self.catch4, self.catch4,axes[1])
            plt.show()
