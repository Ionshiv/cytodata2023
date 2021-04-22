from cnnlayer import CnnLayer


class DataAugmenter:

    def __init__(self):
        self.startstop(True)
        l1 = CnnLayer(1)
        self.startstop(False)
    
    def startstop(self, bool_var: bool):

        if bool_var:
            print("+++INNITIALIZING DATA AUGMENTER+++")
        else:
            print("+++TERMINATING DATA AUGMENTER+++")
        
    

