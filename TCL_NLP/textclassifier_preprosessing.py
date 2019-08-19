imoprt numpy as np
import pandas as pd
import re
import json
from konlpy.tag imoprt Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

DATA_IN_PATH= './data_in/'
train_data = pd.read_csv(DATA_IN_PATH+'ratings_train.txt', header=0, delimiter='\t', quoting=3)
