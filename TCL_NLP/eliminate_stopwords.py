from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

stop_words = "그냥 아무거나 아무렇게나"
stop_words = stop_words.split(' ')


def GetToken(sentence):
    #sentence = "답변 부탁합니다."

    word_tokens = word_tokenize(sentence)
    result = []

    for w in word_tokens:
        if w not in stop_words:
            result.append(w)

    return result

def GetToken(text):
    sentences = []
    vocab = Counter()

    for i in text:
        sentence = word_tokenize(i)
        result = []

        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                if len(word) >2:
                    result.append(word)
                    vocab[word] = vocab[word]+1

        sentences.append(result)



