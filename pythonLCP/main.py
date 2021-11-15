import builtins
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
    
    autoencoder = exp_case(3)
    autoencoder.getSummaryAutoencoder();

def exp_case(args):
    autoencoder = LcpAe();
    switcher = {
        1: exp143ae,
        2: exp147ae,
        3: exp156ae,
        4: exp180ae
    }
    func = switcher[args]
    return func(autoencoder);

def exp143ae(autoencoder:LcpAe):
    print('+++ GENERATING: exp143 +++')
    autoencoder.buildAutoEncoder(24,1080,1080, 6)
    return autoencoder

def exp147ae(autoencoder:LcpAe):
    print('+++ GENERATING: exp147 +++')
    autoencoder.buildAutoEncoder(24,1080,1080, 6)
    return autoencoder

def exp156ae(autoencoder:LcpAe):
    print('+++ GENERATING: exp156 +++')
    autoencoder.buildAutoEncoder(14,1080,1080, 5)
    return autoencoder

def exp180ae(autoencoder:LcpAe):
    print('+++ GENERATING: exp180 +++')
    autoencoder.buildAutoEncoder(24,1080,1080, 6)
    return autoencoder


if __name__ == "__main__":

    main();
