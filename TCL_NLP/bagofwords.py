from konlpy.tag import Okt
import re
okt = Okt()

token = re.sub("(\.)", "","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
token = okt.morphs(token)

word2index = {}
bow=[]
stop_words = {"은", "는", "이", "가", "과", "와", "을", "를"}

for voca in token:
    if voca not in stop_words: # add by me
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)
            bow.insert(len(word2index)-1,1)
        else:
            index = word2index.get(voca)
            bow[index] = bow[index]+1

print(word2index)