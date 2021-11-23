import numpy as np
import glob as glob
class LcpGenerator:

    def __init__(self, inpath:str):
        self.batch_size = 8;
        self.inpath = inpath;
        #format should be /exp1*/
        # self.imagepath = glob.glob(self.inpath + '/sk' + '*')
        self.rcpath = sorted(glob.glob(inpath+'/*'));
        self.validpath = self.rcpath[3::4]
        self.trainpath = [i for i in self.rcpath if i not in self.validpath]

        # print('Full path:       ', self.rcpath, '\n')
        # print('validation path:     ', self.validpath, '\n')
        # print('training path:       ', self.trainpath, '\n')


        print('defining generator')

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
        # print(list2)
        # print(list3)
        print(list1)
        # list1 = iter(list1)
        # print(list1)
        # print(next(list1))
        # print(next(list1))
        Iterating = True
        # while Iterating:
        #     curstr = next(list1, 'last')
        #     print(curstr)
        #     if curstr == 'last':
        #         Iterating = False
        # list1 = list(list1)
        # print(list1)
        while Iterating:
            value = list1.pop(0)
            print(value)
            if not list1:
                Iterating = False


    def getImage(self, impath):
        # print('load images');
        npImage = np.load(impath);
        return npImage;

    def simpleTGen(self):
        # print(self.trainpath);
        trainpath = self.trainpath.copy();
        # print(trainpath)
        while True:
            # if not trainpath:
            #     break
            seqpath = trainpath.pop(0);
            print(seqpath)
            seq = [];
            framepath = sorted(glob.glob(seqpath+'/sk*'));
            for j, imageframe in enumerate(framepath):
                # print(j);
                # print(imageframe);
                frame = self.getImage(imageframe);
                seq += [frame];
            seqx = seq
            seqy = seq
            yield seqx, seqy      

    def simpleVGen(self):
        # print(self.validpath);
        validpath = self.validpath.copy();
        while True:
            # if not validpath:
            #     break
            seqpath = validpath.pop(0);
            seq = [];
            framepath = sorted(glob.glob(seqpath+'/sk*'));
            for j, imageframe in enumerate(framepath):
                print(j);
                print(imageframe);
                frame = self.getImage(imageframe);
                seq += [frame];
            print(seq)
            seqx = seq
            seqy = seq
            yield seqx, seqy