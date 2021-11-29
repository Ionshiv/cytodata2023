import numpy as np
from lcpae import LcpAe
from lcpgenerator import LcpGenerator
import matplotlib.pyplot as plt


def main():
    handshake();
    lcpAutoencoder = buildAutoencoder('exp147', 2);
    lcpAutoencoder.compileAutoencoder();
    # lcpAutoencoder.getSummaryAutoencoder();
    # lcpAutoencoder.testGenerator();
    history = lcpAutoencoder.fitAutoencoder();
    endshake();

def handshake():
    print("+++INNITIALIZING SESSION+++");

def endshake():
    print("+++ENDING SESSION+++");


def buildAutoencoder(case:str, batch_size:int, input_aug:bool=False):
    
    lcpAutoencoder = exp_case(case, batch_size, input_aug=input_aug);
    return lcpAutoencoder

def runAutoencoder(autoencoder:LcpAe):
    print('run')
    history = autoencoder.fitAutoencoder();
    return history


def exp_case(case, batch_size, input_aug:bool=False):
    # lcpAutoencoder = LcpAe();
    switcher = {
        'exp143': exp143ae,
        'exp147': exp147ae,
        'exp156': exp156ae,
        'exp180': exp180ae
    }
    lcpAutoencoder = switcher[case](batch_size, input_aug)
    return lcpAutoencoder;

def exp143ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp143 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp143', batch_size=batch_size)
    lcpAutoencoder = LcpAe(24,1080,1080, 6, inpath='../data/ki-database/exp143', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp147ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp147 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp147', batch_size=batch_size, input_aug=True)
    lcpAutoencoder = LcpAe(25,1080,1080, 6, inpath='../data/ki-database/exp147', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp156ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp156 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp156', batch_size=batch_size)
    lcpAutoencoder = LcpAe(14,1080,1080, 5, inpath='../data/ki-database/exp156', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp180ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp180 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp180', batch_size=batch_size)
    lcpAutoencoder = LcpAe(24,1080,1080, 6, inpath='../data/ki-database/exp180', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder


if __name__ == "__main__":

    main();


