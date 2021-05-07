

class CnnLive:

    def __init__(self):
        self.startstop(True)
        self.startstop(False)

    def startstop(self, bool_var:bool):
        if bool_var:
            print("+++STARTING CNN FOR LIVE CELL PAINTING CLASSIFICATION+++")
        else:
            print("+++ENDING CNN FOR LIVE CELL PAINTING+++")
        
