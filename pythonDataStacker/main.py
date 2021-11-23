from stacker import Stacker
def main():
    print('+++ INNITIATE STACKER +++')
    runStacker();

def runStacker():
    print('+++ EXECUTING STACKER +++')
    stkr = Stacker();
    stkr.makeStacks('all');

if __name__ == "__main__":
    
    main();
