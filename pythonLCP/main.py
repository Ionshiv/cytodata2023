from lcpvae import LcpVae
from stacker import Stacker
import cv2 as cv2

def main():
    handshake()
    # vae = LcpVae()
    runStacker()
    # data = cv2.imread('/home/jovyan/workspace/ki-database/exp143/Images/r05c05f01p01-ch1sk1fk1fl1.tiff', -1)
    # print(data)
    # print(type(data))
    endshake()

def handshake():
    print("+++INNITIALIZING SESSION+++")

def endshake():
    print("+++ENDING SESSION+++")

def runStacker():
    print("+++ EXECUTING STACKER-COMMAND+++")
    Stacker()



if __name__ == "__main__":
    main()
