#https://github.com/e9t/nsmc

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

DATA_IN_PATH = "./data_in"
print("파일크기: ")
for file in os.listdir(DATA_IN_PATH):
  if 'txt' in file:
    print(file.ljust(30) + str(round(os.path.getsize(PATH+file) /1000000,2)) + "MB")

train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header=0, delimiter='\t', quoting=3)
train_data.head()
