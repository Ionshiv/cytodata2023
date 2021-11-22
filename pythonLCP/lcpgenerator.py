import numpy as np
import glob as glob
class LcpGenerator:

    def __init__(self, inpath:str):
        self.batch_size = 8;
        self.inpath = inpath;
        #format should be /exp1*/
        # self.imagepath = glob.glob(self.inpath + '/sk' + '*')
        self.rcpath = sorted(glob.glob(inpath+'/*'));



        print('defining generator')

    def trainGen(self):
        while True:
            print('soon');
            batch_in = [];
            batch_out = [];
            


        

    def validGen(self):

        print('Validation Generator')

    def getImage(self, impath):
        print('load images');
        npImage = np.load(impath);
        return npImage;

    def simpleTGen(self):
        print(self.rcpath);
        for i, seqpath in enumerate(self.rcpath):
            print(i);
            print(seqpath);
            seq = [];
            framepath = sorted(glob.glob(seqpath+'/sk*'));
            for j, jframe in enumerate(framepath):
                print(j);
                print(jframe);
                frame = self.getImage(jframe);
                seq += [frame];
            print(seq)

            
            

    def simpleVGen(self):
        while True:
            print('soon');