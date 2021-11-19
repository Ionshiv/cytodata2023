import numpy as np
class LcpGenerator:

    def __init__(self, inpath:str):
        self.batch_size = 8;
        self.inpath = inpath;


        print('defining generator')

    def trainGen(self):
        while True:
            print(True);
            batch_in = [];
            batch_out = [];
            


        

    def validGen(self):

        print('Validation Generator')

    def getImage(self, rescale:str):
        print('load images');
        npImage = np.load(self.inpath);
        return npImage;