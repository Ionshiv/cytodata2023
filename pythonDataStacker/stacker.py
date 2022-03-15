from lzma import CHECK_CRC64
import numpy as np
import cv2 as cv2
from pathlib import Path




class Stacker:

    def __init__(self):
        print('+++ STACKER INNITIALIZING +++')
        self.inpath_exp143 = '/home/jovyan/workspace/ki-database/exp143/Images/'
        self.inpath_exp147 = '/home/jovyan/workspace/ki-database/exp147/t1/Images/'
        self.inpath_exp147t0 = '/home/jovyan/workspace/ki-database/exp147/t0/Images/'
        self.inpath_exp156 = '/home/jovyan/workspace/ki-database/exp156/Images/'
        self.inpath_exp180 = '/home/jovyan/workspace/ki-database/exp180/Images/'
        self.inpath_exp183 = '/home/jovyan/workspace/ki-database/exp183/Images/'

        self.outpath_exp143 = '/home/jovyan/scratch2-shared/david/liveCellPainting/ki-database/exp143/'
        self.outpath_exp147 = '/home/jovyan/scratch2-shared/david/liveCellPainting/ki-database/exp147/'
        self.outpath_exp156 = '/home/jovyan/scratch2-shared/david/liveCellPainting/ki-database/exp156/'
        self.outpath_exp180 = '/home/jovyan/scratch2-shared/david/liveCellPainting/ki-database/exp180/'
        print('+++ INNITIATING... +++')


    def makeStacks(self, args:str):
        print('+++ GENERATING STACKS: ' + args + ' +++')
        switcher = {
            'exp143': self.run143,
            'exp147': self.run147,
            'exp156': self.run156,
            'exp180_3': self.run180_3,
            'all': self.runAll
        }
        func = switcher[args]
        func()

    def runAll(self):
        print('+++ PROCESSING ALL EXPERIMENTS +++')
        self.run143();
        self.run147();
        # self.run156();
        self.run180_3();
        print('+++ ALL EXPERIMENTS PROCESSED +++')
     
    def run143(self):
        #exp 143 r7-28s1 c3-24s1 ch6
        print('+++++++++++++++++++++++++++++++++ PROCESSING EXP143 ++++++++++++++++++++++++++++++++++++')
        for i in range(5, 29):
            if i < 10:
                rowstr = '0' + str(i)
            else:
                rowstr = str(i)
            for j in range(3, 25, 4):
                for j2 in range (2):
                    if j < 10:
                        colstr = '0'+str(j+j2)
                    else:
                        colstr = str(j+j2)
                    dataSeq = self.makeSequence(self.inpath_exp143 + 'r' + rowstr + 'c' + colstr + 'f01p01-ch',1 , 25, 6)
                    # Stacker.saveim(self.outpath_exp143 + 'r' + rowstr + 'c' + colstr, 'sequence', dataSeq)
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch0', dataSeq[0])
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch1', dataSeq[1])
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch2', dataSeq[2])
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch3', dataSeq[3])
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch4', dataSeq[4])
                    Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch5', dataSeq[5])

    def run147(self):
        #exp 147 r_17-18s1 c3-25s2 ch6
        print('+++++++++++++++ PROCESSING EXP147 +++++++++++++++++++++')
        for i in range(17, 19):
            for j in range(3, 27, 2):
                if j < 10:
                    colstr = '0'+str(j)
                else:
                    colstr = str(j)
                t0Frame = self.makestack(self.inpath_exp147t0 + 'r' + str(i) + 'c' + colstr + 'f01p01-ch', 'sk1fk1fl1.tiff', 6)
                # t0Frame = np.reshape(t0Frame, (1, t0Frame.shape[0], t0Frame.shape[1], 6))
                dataSeq = self.makeSequence(self.inpath_exp147 + 'r' + str(i) + 'c' + colstr + 'f01p01-ch',1 , 25, 6)
                ch0t0 = np.reshape(t0Frame[:,:,0], (1, 1080, 1080))
                ch1t0 = np.reshape(t0Frame[:,:,1], (1, 1080, 1080))
                ch2t0 = np.reshape(t0Frame[:,:,2], (1, 1080, 1080))
                ch3t0 = np.reshape(t0Frame[:,:,3], (1, 1080, 1080))
                ch4t0 = np.reshape(t0Frame[:,:,4], (1, 1080, 1080))
                ch5t0 = np.reshape(t0Frame[:,:,5], (1, 1080, 1080))
                ch0seq = np.concatenate((ch0t0, dataSeq[0]), axis=0)
                ch1seq = np.concatenate((ch1t0, dataSeq[1]), axis=0)
                ch2seq = np.concatenate((ch2t0, dataSeq[2]), axis=0)
                ch3seq = np.concatenate((ch3t0, dataSeq[3]), axis=0)
                ch4seq = np.concatenate((ch4t0, dataSeq[4]), axis=0)
                ch5seq = np.concatenate((ch5t0, dataSeq[5]), axis=0)
                # dataSeq = np.concatenate((t0Frame, dataSeq), 0)
                # Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'sequence', dataSeq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch0', ch0seq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch1', ch1seq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch2', ch2seq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch3', ch3seq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch4', ch4seq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch5', ch5seq)
           
    def run156(self):
        #exp 156 r3-30s1 c3-46s1 ch5
        print('++++++++++++++++++++++++++++++ PROCESSING EXP156 ++++++++++++++++++++++++++++++++++++++++++++')
        for i in range(3, 31):
            if i < 10:
                rowstr = '0' + str(i)
            else:
                rowstr = str(i)
            for j in range(3, 47):
                if j < 10:
                    colstr = '0'+str(j)
                else:
                    colstr = str(j)
                dataSeq = self.makeSequence(self.inpath_exp156 + 'r' + rowstr + 'c' + colstr + 'f01p01-ch',1 , 15, 5)
                Stacker.saveim(self.outpath_exp156 + 'r' + rowstr + 'c' + colstr, 'sequence', dataSeq)

    def run180_3(self):
        #exp 180/3 r3-30s1 c3-46s1 ch6
        print('+++++++++++++++++++++++++++++++++++++++++++++ PROCESSING EXP180 AND EXP183 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for i in range(3, 31):
            if i < 10:
                rowstr = '0' + str(i)
            else:
                rowstr = str(i)
            for j in range(3, 47):
                if j < 10:
                    colstr = '0'+str(j)
                else:
                    colstr = str(j)
                dataSeq = self.repairSequence(self.inpath_exp180 + 'r' + rowstr + 'c' + colstr + 'f01p01-ch', self.inpath_exp183 + 'r' + rowstr + 'c' + colstr + 'f01p01-ch', 1, 24, 1, 3, 6)
                # Stacker.saveim(self.outpath_exp180 + 'r' + rowstr + 'c' + colstr, 'sequence', dataSeq)
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch0', dataSeq[0])
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch1', dataSeq[1])
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch2', dataSeq[2])
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch3', dataSeq[3])
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch4', dataSeq[4])
                Stacker.saveim(self.outpath_exp147 + 'r' + str(i) + 'c' + colstr, 'ch5', dataSeq[5])


    def loadim(inpath: str):
        print('+++ LOADING DATA +++')
        data = cv2.imread(inpath, -1)
        # print(np.amax(data))
        data = data.astype(np.float32)
        data = data/np.amax(data)
        # print(np.amax(data))
        return data

    def saveim(outpath: str, enumstr: str, dataStack):
        print('+++ SAVING STACKED IMAGE +++')
        Path(outpath + '/').mkdir(parents=True, exist_ok=True)
        savepath = outpath + "/" + enumstr
        np.save(savepath, dataStack)
    
    def makestack(self, inpath: str, timestr: str, channels: int):
        print('+++ MAKING STACK... +++')

        if channels == 6:
            print('+++ 6 channels +++')
            data0 = Stacker.loadim(inpath + str(1) + timestr)
            data1 = Stacker.loadim(inpath + str(2) + timestr)
            data2 = Stacker.loadim(inpath + str(3) + timestr)
            data3 = Stacker.loadim(inpath + str(4) + timestr)
            data4 = Stacker.loadim(inpath + str(5) + timestr)
            data5 = Stacker.loadim(inpath + str(6) + timestr)
            dataStack = np.zeros((data0.shape[0], data0.shape[1], 6))
            dataStack[:,:,0] = data0
            dataStack[:,:,1] = data1
            dataStack[:,:,2] = data2
            dataStack[:,:,3] = data3
            dataStack[:,:,4] = data4
            dataStack[:,:,5] = data5
            

        elif channels == 5:
            print('+++ 5 channels +++')
            data0 = Stacker.loadim(inpath + str(1) + timestr)
            data1 = Stacker.loadim(inpath + str(2) + timestr)
            data2 = Stacker.loadim(inpath + str(3) + timestr)
            data3 = Stacker.loadim(inpath + str(4) + timestr)
            data4 = Stacker.loadim(inpath + str(5) + timestr)
            dataStack = np.zeros((data0.shape[0], data0.shape[1], 5))
            dataStack[:,:,0] = data0
            dataStack[:,:,1] = data1
            dataStack[:,:,2] = data2
            dataStack[:,:,3] = data3
            dataStack[:,:,4] = data4

        else:
            print('+++FEATURE NOT SUPPORTED+++')

        return dataStack

    def makeSequence(self, inpath:str, timestart:int, timeend:int, channels:int):
        print('making sequence')
        dataSequence = [];
        ch0seq = []
        ch1seq = []
        ch2seq = []
        ch3seq = []
        ch4seq = []
        ch5seq = []
        for k in range(timestart, timeend):
            dataFrame =  self.makestack(inpath, 'sk' + str(k) + 'fk1fl1.tiff', channels )
            # dataFrame = np.reshape(dataFrame, (1, dataFrame.shape[0], dataFrame.shape[1], channels))
            ch0frame = dataFrame[:,:,0]
            ch0frame = np.reshape(ch0frame, (1, 1080, 1080))
            ch1frame = dataFrame[:,:,1]
            ch1frame = np.reshape(ch1frame, (1, 1080, 1080))
            ch2frame = dataFrame[:,:,2]
            ch2frame = np.reshape(ch2frame, (1, 1080, 1080))
            ch3frame = dataFrame[:,:,3]
            ch3frame = np.reshape(ch3frame, (1, 1080, 1080))
            ch4frame = dataFrame[:,:,4]
            ch4frame = np.reshape(ch4frame, (1, 1080, 1080))
            ch5frame = dataFrame[:,:,5]
            ch5frame = np.reshape(ch5frame, (1, 1080, 1080))
            if k == timestart:
                # dataSequence = dataFrame;
                ch0seq = ch0frame
                ch1seq = ch1frame
                ch2seq = ch2frame
                ch3seq = ch3frame
                ch4seq = ch4frame
                ch5seq = ch5frame
            else:
                # dataSequence = np.concatenate((dataSequence, dataFrame), 0)
                ch0seq = np.concatenate((ch0seq, ch0frame), axis=0)
                ch1seq = np.concatenate((ch1seq, ch1frame), axis=0)
                ch2seq = np.concatenate((ch2seq, ch2frame), axis=0)
                ch3seq = np.concatenate((ch3seq, ch3frame), axis=0)
                ch4seq = np.concatenate((ch4seq, ch4frame), axis=0)
                ch5seq = np.concatenate((ch5seq, ch5frame), axis=0)
        dataSequence = [ch0seq] + [ch1seq] + [ch2seq] + [ch3seq] + [ch4seq] + [ch5seq]
        return dataSequence
    
    def repairSequence(self, inpath1:str, inpath2:str, tstart1:int, tend1:int, tstart2:int, tend2:int, channels:int):
        seq180 = self.makeSequence(inpath=inpath1, timestart=tstart1, timeend=tend1, channels=channels)
        seq183 = self.makeSequence(inpath=inpath2, timestart=tstart2, timeend=tend2, channels=channels)
        ch0 = np.concatenate((seq180[0],seq183[0]), 0)
        ch1 = np.concatenate((seq180[1],seq183[1]), 0)
        ch2 = np.concatenate((seq180[2],seq183[2]), 0)
        ch3 = np.concatenate((seq180[3],seq183[3]), 0)
        ch4 = np.concatenate((seq180[4],seq183[4]), 0)
        ch5 = np.concatenate((seq180[5],seq183[5]), 0)
        dataSequence = [ch0] + [ch1] + [ch2] + [ch3] + [ch4] + [ch5]
        return dataSequence

