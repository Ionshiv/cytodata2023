from keras.backend import reshape
import numpy as np
import glob as glob
class LcpGenerator:

    def __init__(self, inpath:str, batch_size:int):
        self.batch_size = batch_size;
        self.inpath = inpath;
        self.rcpath = sorted(glob.glob(inpath+'/*'));
        self.validpath = self.rcpath[3::4]
        self.trainpath = [i for i in self.rcpath if i not in self.validpath]
        self.steps_batch = int(len(self.trainpath)/self.batch_size)
        print('length of train', len(self.trainpath))
        # print(self.trainpath)

    def trainGen(self):
        while True:
            print('soon');
            batch_in = [];
            batch_out = [];

    def validGen(self):

        print('Validation Generator')

    def testcase(self):
        list1 = ['qwe', 'rty', 'uio', 'pas', 'dfg', 'hjk', 'lzx', 'cvb', 'nm', 'åöä']
        # list2 = [ 'rty', 'pas', 'hjk']
        list2 = list1[2::3]
        list3 = [i for i in list1 if i not in list2]
        print(list1)
        Iterating = True
        while Iterating:
            value = list1.pop(0)
            print(value)
            if not list1:
                Iterating = False


    def getImage(self, impath):
        npImage = np.load(impath);
        npImage = npImage/255;
        # print(type(npImage))
        npImage = np.reshape(npImage, (1, npImage.shape[0], npImage.shape[1], npImage.shape[2], npImage.shape[3]))
        return npImage;

    def simpleTGen(self):
        # print('TESTING')
        trainpath = self.trainpath.copy();
        # print(trainpath)
        while True:
            if not trainpath:
                break
            seqpath = trainpath.pop(0);
            # print(seqpath)
            seq = self.getImage(seqpath + '/sequence.npy')
            # print('this is input shape: ', seq.shape)
            seqx = seq
            seqy = seq
            yield seqx, seqy      

    def simpleVGen(self):
        validpath = self.validpath.copy();
        while True:
            if not validpath:
                break
            seqpath = validpath.pop(0);
            seq = self.getImage(seqpath + '/sequence.npy')
            seqx = seq
            seqy = seq
            yield seqx, seqy