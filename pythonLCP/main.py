from lcpae import LcpAe
from stacker import Stacker
import cv2 as cv2

def main():
    handshake();
    runAutoencoder();
    # runStacker();
    endshake();

def handshake():
    print("+++INNITIALIZING SESSION+++");

def endshake():
    print("+++ENDING SESSION+++");

def runStacker():
    print("+++ EXECUTING STACKER-COMMAND+++");
    Stacker();

def runAutoencoder():
    crae = LcpAe();
    crae.getSummaryAutoencoder();

if __name__ == "__main__":

    main();
