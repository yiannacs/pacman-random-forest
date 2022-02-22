# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np

class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def fit(self, data, target):
        data = self.parseData(data)
        print(data.shape)

    def predict(self, data, legal=None):
        return 1
    
    def parseData(self, data):
        # print('ruifhceruifhceuij')
        data = np.array(data)
        for i in range(len(data)):
            # val = 1
            walls = np.zeros((len(data)))
            for j in range(0, 4):
                walls = (walls*2) + data[:,j]
                
            food = np.zeros((len(data)))
            for j in range(4, 8):
                food = (food*2) + data[:,j]
                
            ghost1 = np.zeros((len(data)))
            for j in range(8, 16):
                ghost1 = (ghost1*2) + data[:,j]
                
            ghost2 = np.zeros((len(data)))
            for j in range(16, 24):
                ghost2 = (ghost2*2) + data[:,j]
                
            ghostVisible = np.zeros((len(data)))
            ghostVisible = ghostVisible + data[:,24]        
        
        return np.array([walls, food, ghost1, ghost2, ghostVisible]).T
        
        
