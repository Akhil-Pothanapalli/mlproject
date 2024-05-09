import os
import sys

import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) #to save object we need a file path

        os.makedirs(dir_path, exist_ok= True) #there should a directory to store if not create it

        with open(file_path, "wb") as file_obj: # we will open the file at the file_path as "wb" wirte binary ,ode
            
            pickle.dump(obj, file_obj)   # dump function will serialize the python object "obj"
                                        #serialization converts the object into byte stream and will be writeen into file_obj

    except Exception as e:
        raise CustomException(e, sys)