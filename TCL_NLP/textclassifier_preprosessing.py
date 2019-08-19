imoprt numpy as np
import pandas as pd
import re
import json
from konlpy.tag imoprt Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

#const & get data from csv
DATA_IN_PATH= './data_in/'
MAX_SEQUENCE_LENGTH = 8 #단어의 평균 개수
train_data = pd.read_csv(DATA_IN_PATH+'ratings_train.txt', header=0, delimiter='\t', quoting=3)
#train_data['document'][:5]

#initialize
review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", train['document'][0])
okt = Okt()
review_text = okt.morphs(review_text, stem=True)
stop_words = set(['은','는','이','가','하','아','것','들','의','있', '되','수','보','주', '등','한','을','를','과','와'])
clean_review= [token for w in review if not token in stop_words]

#preprocessing
def preprocessing(review, okt, remove_stopwords = False, stop_words=[]):
  review_text = re.sub("[가-힣ㄱ-하ㅏ-ㅣ\\s]", "", review)
  word_review = okt.morphs(review_text, stem=True)
  if remove_stopwords:
    word_review = [token for token in word_review if not token in stop_words]
  return word_review

#clean_train_review
clean_train_review= []
for review in train_data['document']:
  if type(review) == str:
    clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
  else:
    clean_train_review.append([])

#clean_test_review
test_data = pd.read_csv(DATA_IN_PATH+'ratings_test.txt', header=0, delimiter='\t', quoting=3)
clean_test_review = []
for review in test_data['document']:
 if type(review)==str:
  clean_test_review.append(preprocessing(review,okt,remove_stopwords=True, stop_words=stop_words))
 else:
  clean_test_review.append([])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)
word_vocab = tokenizer.word_index
train_inputs = pad_sequences(train_sequeces, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(train_data['label'])
test_inputs = pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(test_data['label'])

TRAIN_INPUT_DATA ='nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TRAIN_INPUT_DATA = 'nsmc_test_input.npy'
TRAIN_LABEL_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'
data_configs = {}

data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)+1

import os
if not os.path.exists(DEFAULT_PATH +DATA_IN_PATH):
  os.makedirs(DEFAULT_PATH+DATA_IN_PATH)

np.save(open(DEFAULT_PATH + DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DEFAULT_PATH + DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)
np.save(open(DEFAULT_PATH + DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DEFAULT_PATH + DATA_IN_PATH + TEST_LABEL_DATA, 'wb'), test_labels)
# 데이터 사전을 json으로 저장
json.dump(data_configs, open(DEFAULT_PATH + DATA_IN_PATH +DATA_CONFIGS, 'w'), ensure_ascii=False)




