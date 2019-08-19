#https://github.com/e9t/nsmc

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

DATA_IN_PATH = "./data_in"
#print("파일크기: ")
#for file in os.listdir(DATA_IN_PATH):
#  if 'txt' in file:
#    print(file.ljust(30) + str(round(os.path.getsize(PATH+file) /1000000,2)) + "MB")

train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header=0, delimiter='\t', quoting=3)
#train_data.head()
train_length = train_data['document'].astype(str).apply(len)
train_review = [review for review in train_data['document'] if type(review) is str]
wordcloud = WordCloud(font_path=DATA_IN_PATH+'NanumGothic.ttf').generate(' '.join(train_review))

#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.show()

train_word_counts = train_data['document'].atype(str).apply(lambda x:len(x.split(' ')))
qmarks = np.mean(train_data['document'].astype(str).apply(lambda x: '?' in x))
fullstop = np.mean(train_data['document'].astype(str).apply(lambda x: '.' in x))


