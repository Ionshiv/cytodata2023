
from livegan import LiveGan
from livecnn import LiveCnn

def main():
    handshake()
    dgan = LiveGan()
    cnnl = LiveCnn()
    endshake()

def handshake():
    print("+++HELLO USER. INNITIALIZING SESSION+++")

def endshake():
    print("+++ENDING SESSION+++")


if __name__ == "__main__":
    main()
