"""
Show how to use a lasso to select a set of points and get the indices
of the selected points.  A callback is used to change the color of the
selected points

This is currently a proof-of-concept implementation (though it is
usable as is).  There will be some refinement of the API.
"""
from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path
import matplotlib.image as mpimg
from simplebar import barchart as Bar

import matplotlib.pyplot as plt
from numpy import nonzero
from numpy.random import rand
import logging

class Datum(object):
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('black')

    def __init__(self, x, y, labour, include=False):
        self.x = x
        self.y = y
        self.labour = labour

        if include:
            self.color = self.colorin
        else:
            self.color = self.colorout


class LassoManager(object):
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        facecolors = [d.color for d in data]

        # print facecolors

        self.xys = [(d.x, d.y) for d in data]
        fig = ax.figure
        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(20,),
            facecolors=facecolors,
            offsets=self.xys,
            transOffset=ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = Datum.colorin
                self.data[i].color = Datum.colorin
            else:
                facecolors[i] = Datum.colorout
                self.data[i].color = Datum.colorout

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        # print facecolors
        del self.lasso

    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    # def producebar(self, event):
    #
    #     logging.debug('in LassoManager.onKeyPress(). Event received: %s (key: %s)' % (event, event.key))
    #     if event.key == 'x':
    #         bararr =[]
    #         for i in range(len(data)):
    #             if data[i].color == data[i].colorin:
    #                 bararr.append(data[i].labour)
    #
    #         bc = Bar(names1 = bararr, names2 = bararr)
    #         bc.process()
    #         bc.process(grouping = 'Yes')
    #         plt.show()

