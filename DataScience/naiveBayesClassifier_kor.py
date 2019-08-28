from konlpy.tag import Twitter
pos_tagger = Twitter()

train= [('�޸��� ����', 'pos'),('����̵� ����', 'pos'),('�� ������ ������', 'neg'),('�޸��� �̻� �����', 'pos'),('�� ��ġ�� �޸��� ��ž�', 'pos')]
all_words = set(word.lower() for sentene in trian for word in word_tokenize(sentence[0]))

t= [({word: (word in word_tokenize(x[0])) for word in all_words}, BaseException[1]) for x in train]
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

test_sentence = '�� ������ ��ġ�� �޸��� ��ž�'
test_sent_features = {word.lower(): (word in word_tokenize(test_sentence.lower())) for word in all_words}
classifier.classify(test_sent_features)

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

train_docs = [(tokenize(row[0]), row[1]) for row in train]
tokens = [t for d in train_docs for t in d[0]]

def  term_exists(doc):
    return {word: (word in set(doc)) for word in tokens}

train_xy = [(term_exists(d), c) for d,c in train_docs]
classifier = nltk.NaiveBayesClassifier.train(train_xy)

test_sentence = '�� ������ ��ġ�� �޸��� ��ž�'

test_docs = pos_tagger.pos(test_sentence[0])
test_sent_features = {word: (word in tokens) for word in test_docs}
classifier.classify(test_sent_features)


