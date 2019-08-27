from nltk.tokenize import word_tonenize
import nltk

train= [('i like you', 'pos'),
        ('i hate you', 'neg'),
        ('you like me', 'neg'),
        ('i like her', 'pos')]

all_words = set(word.lower() for sentence in train for word in word_tonenize(sentence[0]))
t = [({word: (word in word_tonenize(x[0])) for word in all_words}, x[1]) for x in train]
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

# test
test_sentence = 'i like Merui'
test_sent_features = {word.lower(): (word in word_tonenize(test_sentence.lower())) for word in all_words}

classifier.classify(test_sent_features)


