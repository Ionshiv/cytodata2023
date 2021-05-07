
from dataaugmenter import DataAugmenter
from cnnlive import CnnLive

def main():
    handshake()
    dgan = DataAugmenter()
    cnnl = CnnLive()
    endshake()

def handshake():
    print("+++HELLO USER. INNITIALIZING SESSION+++")

def endshake():
    print("+++ENDING SESSION+++")


if __name__ == "__main__":
    main()
