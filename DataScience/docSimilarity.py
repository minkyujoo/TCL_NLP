from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Twitter
t = Twitter()

vectorizer = CountVectorizer(min_df = 1)

contents = ['�޸��� ����� ������ �ٻ۵� �����?', '�޸��� �������� ��å�ϰ� ��� ���� �Ⱦ��ؿ�', 
            '�޸��� �������� ��� �͵� �Ⱦ��ؿ�. �̻��ؿ�', '�� ������ ������ ������ ������ �ʹ� �ٺ��� �׷��� ���ϰ� �־��']

X = vectorizer.fit_transform(contents)
vectorizer.get_feature_names()

contents_tokens = [t.morphs(row) for row in contents]
contents_for_vectorize = []

for content in contents_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word
    contents_for_vectorize.append(sentence)

X = vectorizer.fit_transform(contents_for_vectorize)
num_samples, num_features = X.shape
X.toarray().transpose()

new_post = ['�޸��� �������� ��å�ϰ� ��� �;��.']
new_post_tokens = [t.morphs(row) for row in new_post]
new_post_for_vectorize = []

for content in new_post_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word

    new_post_for_vectorize.append(sentence)

new_post_vec = vectorizer.transform(new_post_for_vectorize)
new_post_vec.toarray()

import scipy as sp
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = 65535
best_i = None

for i in range(0, num_samples):
    post_vec = X.getrow(i)
    d = dist_raw(post_vec, new_post_vec)

    print("== Post %i with dist=%.2f : %s" % (i,d,contents[i]))
    if d < best_dist:
        best_dist = d
        best_i = i

def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = 65535
best_i = None

for i in range(0, num_samples):
    post_vec = X.getrow(i)
    d = dist_raw(post_vec, new_post_vec)

    print("== Post %i with dist=%.2f : %s" % (i,d,contents[i]))
    if d < best_dist:
        best_dist = d
        best_i = i

# tf-idf (term frequency inverse document frequency)
def tfidf(t,d, D):
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log(float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf, idf

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')

contents_tokens = [t.morphs(row) for row in contents]
contents_for_vectorize = []
for content in contents_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word
    contents_for_vectorize.append(sentence)
X = vectorizer.fit_transform(contents_for_vectorize)
num_samples , num_features = X.shape

vectorizer.get_feature_names()


new_post = ['�޸��� �������� ��å�ϰ� ��� �;��.']
new_post_tokens = [t.morphs(row) for row in new_post]
new_post_for_vectorize = []

for content in new_post_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word

    new_post_for_vectorize.append(sentence)

new_post_vec = vectorizer.transform(new_post_for_vectorize)


best_doc = None
best_dist = 65535
best_i = None

for i in range(0, num_samples):
    post_vec = X.getrow(i)
    d = dist_norm(post_vec, new_post_vec)

    print("== Post %i with dist=%.2f : %s" % (i,d,contents[i]))
    if d < best_dist:
        best_dist = d
        best_i = i
