

class CnnLayer:

    def __init__(self, dimension):
        self.msg(True)
        self.msg(False)
    
    def msg(self, bvar: bool):
        if bvar:
            print("+++DEFINING CNN LAYER+++")
        else:
            print("+++LAYER DEFINED+++")


