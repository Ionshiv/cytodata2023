from keras.backend import reshape
import numpy as np
import glob as glob
class LcpGenerator:

    def __init__(self, inpath:str, batch_size:int=1, input_aug:bool=False):
        self.input_aug = input_aug;
        self.batch_size = batch_size;
        self.inpath = inpath;
        self.rcpath = sorted(glob.glob(inpath+'/*'));
        self.validpath = self.rcpath[3::4]
        self.trainpath = [i for i in self.rcpath if i not in self.validpath]
        self.steps_batch = len(self.trainpath)//self.batch_size
        print('length of train', len(self.trainpath))

    def trainGen(self):
        while True:
            batch = [];
            batch_files = np.random.choice(self.trainpath, size = self.batch_size);
            for i, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch += [seq]
                if self.input_aug:
                    seq_aug = self.augment(seq);
                    batch += seq_aug;
            batch_x = np.array(batch);
            batch_y = np.array(batch.copy())
            yield batch_x, batch_y 

    def validGen(self):
        while True:
            batch = [];
            batch_files = np.random.choice(self.validpath, size = self.batch_size);
            for i, seqpath in enumerate(batch_files):
                seq = self.getImage(seqpath + '/sequence.npy')
                batch += [seq]
            batch_x = np.array(batch)
            batch_y = np.array(batch.copy())
            yield batch_x, batch_y
    
    def augment(self, seq):
        seq_aug = []
        seq0_5pi = np.rot90(seq, k=1, axes=(1, 2))
        seq1pi = np.rot90(seq, k=2, axes=(1, 2))
        seq1_5pi = np.rot90(seq, k=3, axes=(1, 2))
        seq_aug += [seq0_5pi];
        seq_aug += [seq1pi];
        seq_aug += [seq1_5pi];
        return seq_aug

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
        # npImage = np.reshape(npImage, (1, npImage.shape[0], npImage.shape[1], npImage.shape[2], npImage.shape[3]))
        return npImage;

    def simpleTGen(self):
        while True:
            seqpath = np.random.choice(self.trainpath, size = 1)
            seq = self.getImage(seqpath[0] + '/sequence.npy')
            seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1], seq.shape[2], seq.shape[3]))
            seqx = seq
            seqy = seq
            yield seqx, seqy      

    def simpleVGen(self):
        while True:
            seqpath = np.random.choice(self.validpath, size=1)
            seq = self.getImage(seqpath[0] + '/sequence.npy')
            seq = np.reshape(seq, (1, seq.shape[0], seq.shape[1], seq.shape[2], seq.shape[3]))
            seqx = seq
            seqy = seq
            yield seqx, seqy