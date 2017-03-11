'''
Code for classifying the category of a question on Jeopardy!
'''

import string
from collections import defaultdict
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

stopwords = set(stopwords.words("english"))
punctuation = set(string.punctuation)

def parseData():
    with open('jeopardy.json') as data_file:
        data = data_file.read()
        data = string.replace(data, 'null', 'None')
        return eval(data)

data = parseData()
#print parseData()[:2]

def wordCount(data):
    word_count = defaultdict(int)
    for d in data:
        words = ''.join([c for c in d['question'].lower() if not c in punctuation]).split()
        for w in words: # and if w not in stopwords:
            word_count[w] += 1
    word_count = [(word_count[w], w) for w in word_count]
    word_count.sort()
    word_count.reverse()
    return word_count[:1000]   # return last-however-many here

def categoryCount(data):
    category_counts = defaultdict(int)
    for d in data:
        category_counts[d['category']] += 1

    category_counts = [(category_counts[k], k) for k in category_counts]
    category_counts.sort()
    category_counts.reverse()
    #print "Number of categories: {}\nTop 10 categories: {}".format(len(category_counts), category_counts[:10])
    #print "Mean number of items per categories: {}\nStd dev of items per categories: {}".format(np.mean([x[0] for x in category_counts]), np.std([x[0] for x in category_counts]))
    return category_counts


word_counts = wordCount(data)
category_counts = categoryCount(data)
word_id = {x[1]: num for (x, num) in zip(word_counts, range(len(word_counts)))}
category_id = {x[1]: num for (x, num) in zip(category_counts, range(len(category_counts)))}
#print len(wordCount(data))

#train_raw_data = data[:(len(data)/2)]
#val_raw_data = data[(len(data)/2):]

X = []
y = []
for d in data:
    x = [0] * len(word_id)
    words = ''.join([c for c in d['question'].lower() if not c in punctuation]).split()
    for w in words:
        if w in word_id:
            x[word_id[w]] = 1
    X.append(x)
    y.append(category_id[d['category']])

print "Done loading data!"

X_train, X_val = np.asmatrix(X[:(len(X)/2)]), np.asmatrix(X[(len(X)/2):])
y_train, y_val = np.asarray(y[:(len(y)/2)]), np.asarray(y[(len(y)/2):])

model = LinearSVC(C=1)
model.fit(X_train, y_train)
print "Score: {}".format(model.score(X_val, y_val))
