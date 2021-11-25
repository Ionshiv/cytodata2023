import numpy as np
from lcpae import LcpAe
from lcpgenerator import LcpGenerator
import matplotlib.pyplot as plt


def main():
    handshake();
    lcpAutoencoder, lcpGen = buildAutoencoder();
    lcpAutoencoder.compileAutoencoder();
    # lcpAutoencoder.getSummaryAutoencoder();
    history = lcpAutoencoder.fitAutoencoder(lcpGen=lcpGen);
    # print(type(History))
    # A = np.load('../data/ki-database/exp147/r17c03/sequence.npy')
    # B = np.load('../data/ki-database/exp143/r08c03/sequence.npy')
    # C = np.load('../data/ki-database/exp156/r17c03/sequence.npy')
    # D = np.load('../data/ki-database/exp180/r17c03/sequence.npy')
    # print(A.shape, '/',B.shape,'/', C.shape,'/', D.shape)
    # print(A[1, :, :, :])
    # A, B = next(lcpGen.simpleTGen())
    # print(A.shape, B.shape)
    endshake();

def handshake():
    print("+++INNITIALIZING SESSION+++");

def endshake():
    print("+++ENDING SESSION+++");


def buildAutoencoder():
    
    # autoencoder, seqGen = exp_case(1);
    lcpAutoencoder, lcpGen = exp_case(1);
    # exp_case(1)
    # autoencoder.getSummaryAutoencoder();
    return lcpAutoencoder, lcpGen

def runAutoencoder(autoencoder:LcpAe, lcpGen:LcpGenerator):
    print('run')
    history = autoencoder.fitAutoencoder(lcpGen=lcpGen)
    return history


def exp_case(args):
    # lcpAutoencoder = LcpAe();
    switcher = {
        0: exp143ae,
        1: exp147ae,
        2: exp156ae,
        3: exp180ae
    }
    lcpAutoencoder, lcpGen = switcher[args]()
    return lcpAutoencoder, lcpGen;

def exp143ae():
    print('+++ GENERATING: exp143 +++')
    lcpGen = LcpGenerator(inpath='../data/ki-database/exp143')
    lcpAutoencoder = LcpAe(24,1080,1080, 6)
    return lcpAutoencoder, lcpGen

def exp147ae():
    print('+++ GENERATING: exp147 +++')
    lcpGen = LcpGenerator(inpath='../data/ki-database/exp147')
    lcpAutoencoder = LcpAe(25,1080,1080, 6)
    return lcpAutoencoder, lcpGen

def exp156ae():
    print('+++ GENERATING: exp156 +++')
    lcpGen = LcpGenerator(inpath='../data/ki-database/exp156')
    lcpAutoencoder = LcpAe(14,1080,1080, 5)
    return lcpAutoencoder, lcpGen

def exp180ae():
    print('+++ GENERATING: exp180 +++')
    lcpGen = LcpGenerator(inpath='../data/ki-database/exp180')
    lcpAutoencoder = LcpAe(24,1080,1080, 6)
    return lcpAutoencoder, lcpGen


if __name__ == "__main__":

    main();


