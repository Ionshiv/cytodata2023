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
        self.steps_batch = len(self.trainpath)//self.batch_size
        print('length of train', len(self.trainpath))
        # print(self.trainpath)

    def trainGen(self):
        while True:
            batch_x = [];
            batch_y = [];
            batch_files = np.random.choice(self.trainpath, size = 8);
            for i, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch_x += [seq]
                print(i)
            for j, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch_y += [seq]
                print(j)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y 

    def validGen(self):

        while True:
            batch_x = [];
            batch_y = [];
            batch_files = np.random.choice(self.validpath, size = 8);
            for i, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch_x += [seq]
                print(i)
            for j, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch_y += [seq]
                print(j)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y 

    def testcase(self):
        list1 = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'hhh', 'iii', 'jjj']
        # list2 = [ 'rty', 'pas', 'hjk']
        # list2 = list1[2::3]
        # list3 = [i for i in list1 if i not in list2]
        # print(list1)
        # Iterating = True
        j = 0;
        while True:
            print('list1 before pop + append:   ', list1)
            value = list1.pop(0)
            list1.append(value)
            print('list1 after pop + append:    ', list1)
            print(value)
            j += 1;
            return value, j
            # if not list1:
            #     Iterating = False


    def getImage(self, impath):
        npImage = np.load(impath);
        npImage = npImage/255;
        # print(type(npImage))
        npImage = np.reshape(npImage, (1, npImage.shape[0], npImage.shape[1], npImage.shape[2], npImage.shape[3]))
        return npImage;

    def simpleTGen(self):
        while True:
            seqpath = np.random.choice(self.trainpath, size = 1)
            seq = self.getImage(seqpath[0] + '/sequence.npy')
            seqx = seq
            seqy = seq
            yield seqx, seqy      

    def simpleVGen(self):
        while True:
            seqpath = np.random.choice(self.validpath, size=1)
            seq = self.getImage(seqpath[0] + '/sequence.npy')
            seqx = seq
            seqy = seq
            yield seqx, seqy