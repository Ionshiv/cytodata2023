import numpy as np
import PIL as pil
from PIL import Image
import cv2 as cv2
from pathlib import Path




class Stacker:

    def __init__(self):
        print('+++ STACKER INNITIALIZING +++')
        inpath_exp147 = '/home/jovyan/workspace/ki-database/exp147/t1/Images/'
        inpath_exp147t0 = '/home/jovyan/workspace/ki-database/exp147/t0/Images/'
        inpath_exp143 = '/home/jovyan/workspace/ki-database/exp143/Images/'

        outpath_exp147 = '/home/jovyan/scratch-shared/david/data/ki-database/exp147/'
        outpath_exp143 = '/home/jovyan/scratch-shared/david/data/ki-database/exp143/'

        #exp 147 r_17-18s1 c3-25s2 ch6
        # for i in range(17, 19):
        #     for j in range(3, 27, 2):
        #         for k in range(2, 25):
        #             if j < 10:
        #                 colstr = '0'+str(j)
        #             else:
        #                 colstr = str(j) 
        #             dataStack =  Stacker.makestack(inpath_exp147 + 'r' + str(i) + 'c' + colstr + 'f01p01-ch', 'sk' + str(k) + 'fk1fl1.tiff', 6 )
        #             Stacker.saveim(outpath_exp147 + 'r' + str(i) + 'c' + colstr + 'f01p01', 'sk' + str(k), dataStack)
        # for i in range(17, 19):
        #     for j in range(3, 27, 2):
        #         if j < 10:
        #             colstr = '0'+str(j)
        #         else:
        #             colstr = str(j) 
        #         dataStack =  Stacker.makestack(inpath_exp147t0 + 'r' + str(i) + 'c' + colstr + 'f01p01-ch', 'sk1fk1fl1.tiff', 6 )
        #         Stacker.saveim(outpath_exp147 + 'r' + str(i) + 'c' + colstr + 'f01p01', 'sk1', dataStack)
        
        #exp 143 r7-28s1 c3-24s1 ch6
        for i in range(5, 29):
            if i < 10:
                rowstr = '0' + str(i)
            else:
                rowstr = str(i)
            for j in range(3, 25, 4):
                for j2 in range (2):
                    for k in range(1, 20):
                        if j < 10:
                            colstr = '0'+str(j+j2)
                        else:
                            colstr = str(j+j2) 
                        dataStack =  Stacker.makestack(inpath_exp143 + 'r' + rowstr + 'c' + colstr + 'f01p01-ch', 'sk' + str(k) + 'fk1fl1.tiff', 6 )
                        Stacker.saveim(outpath_exp143 + 'r' + str(i) + 'c' + str(j+j2) + 'f01p01', 'sk' + str(k), dataStack)


    def loadim(inpath: str):
        print('+++ LOADING DATA +++')
        # image = Image.open(inpath)
        # image.load()
        # data = np.array(image, "int32" )
        data = cv2.imread(inpath, -1)
        return data

    def saveim(outpath: str, enumstr: str, dataStack):
        print('+++ SAVING STACKED IMAGE +++')
        Path(outpath + '/').mkdir(parents=True, exist_ok=True)
        savepath = outpath + "/" + enumstr
        np.save(savepath, dataStack)
    
    def makestack(inpath: str, timestr: str, channels: int):
        print('+++ MAKING STACK... +++')

        if channels == 6:
            print('+++ 6 channels +++')
            data0 = Stacker.loadim(inpath + str(1) + timestr)
            print(inpath + str(1) + timestr) 
            print(type(data0))
            data1 = Stacker.loadim(inpath + str(2) + timestr)
            data2 = Stacker.loadim(inpath + str(3) + timestr)
            data3 = Stacker.loadim(inpath + str(4) + timestr)
            data4 = Stacker.loadim(inpath + str(5) + timestr)
            data5 = Stacker.loadim(inpath + str(6) + timestr)
            dataStack = np.zeros((data0.shape[0], data0.shape[1], 6))
            dataStack[:,:,0] = data0
            dataStack[:,:,0] = data1
            dataStack[:,:,0] = data2
            dataStack[:,:,0] = data3
            dataStack[:,:,0] = data4
            dataStack[:,:,0] = data5

        elif channels == 5:
            print('+++ 5 channels +++')
            data0 = Stacker.loadim(inpath + str(1) + timestr)
            data1 = Stacker.loadim(inpath + str(2) + timestr)
            data2 = Stacker.loadim(inpath + str(3) + timestr)
            data3 = Stacker.loadim(inpath + str(4) + timestr)
            data4 = Stacker.loadim(inpath + str(5) + timestr)
            dataStack = np.zeros((data0.shape(0), data0.shape(1), 5))
            dataStack[:,:,0] = data0
            dataStack[:,:,0] = data1
            dataStack[:,:,0] = data2
            dataStack[:,:,0] = data3
            dataStack[:,:,0] = data4

        else:
            print('+++FEATURE NOT SUPPORTED+++')

        return dataStack
        






