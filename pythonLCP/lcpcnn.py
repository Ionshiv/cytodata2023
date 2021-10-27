import numpy as np
import tensorflow as tf
import pandas as pd
import IPython
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




class LcpCnn:

    def __init__(self):
        self.startstop(True)
        self.startstop(False)

    def startstop(self, bool_var:bool):
        if bool_var:
            print("+++STARTING CNN FOR LIVE CELL PAINTING CLASSIFICATION+++")
        else:
            print("+++ENDING CNN FOR LIVE CELL PAINTING+++")
        
