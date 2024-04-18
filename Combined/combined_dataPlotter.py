import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class DataPlotter(object):
    def __init__(self, title, xlabel, ylabel):
        self.x = xlabel
        self.y = ylabel
        self.title = title
        '''each entry in data is a generation of values [[],[]]'''
        self.data = []
        
    def add_data(self, data):
        self.data.append(data)

    def display_data(self):
        print(matplotlib.get_backend())
        #matplotlib.use('Qt5Agg')
        #import matplotlib
        matplotlib.use('Agg')
        print(matplotlib.get_backend())
        min = []
        mean = []
        max = []
        gen = []
        i = 0
        for generation in self.data:
            min.append(np.min(generation))
            mean.append(np.mean(generation))
            max.append(np.max(generation))
            gen.append(i)
            i += 1

        fig = plt.figure(self.title)

        plt.plot(gen, min, 'b-', label="Minimum")
        plt.plot(gen, mean, 'g-', label="Mean")
        plt.plot(gen, max, 'r-', label="Max")
        
        plt.title(self.title)
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        plt.grid()
        plt.legend(loc="best")
        plt.savefig(f"{self.title}.svg")
        #plt.switch_backend('Qt5Agg')
        #fig.show()
