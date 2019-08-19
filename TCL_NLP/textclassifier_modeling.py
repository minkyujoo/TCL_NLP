import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split

DATA_IN_PATH ='./data_in/'
DATA_OUT_PATH = './data_out/'
INPUT_TRAIN_DATA_FILE_NAME = 'nsmc_train_input.py'
