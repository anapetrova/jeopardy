'''
Code for classifying the category of a question on Jeopardy!
'''

import string
from collections import defaultdict
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import sys


stopwords = set(stopwords.words("english"))
punctuation = set(string.punctuation)
print("starting")
sys.stdout.flush()
def parseData():
    with open('jeopardy.json') as data_file:
        data = data_file.read()
        data = string.replace(data, 'null', 'None')
        return eval(data)

data = parseData()
#print parseData()[:2]
print("DATA PARSED")
sys.stdout.flush()

#gets total word counts
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

#gets total category counts
def categoryCount(data):
    category_counts = defaultdict(int)
    wordFreq = defaultdict(float)
    wordFreqCat = {}
    for d in data:
        category_counts[d['category']] += 1

    category_counts = [(category_counts[k], k) for k in category_counts]
    category_counts.sort()
    category_counts.reverse()
    #print "Number of categories: {}\nTop 10 categories: {}".format(len(category_counts), category_counts[:10])
    #print "Mean number of items per categories: {}\nStd dev of items per categories: {}".format(np.mean([x[0] for x in category_counts]), np.std([x[0] for x in category_counts]))

    return category_counts[:50]

#gets total value counts
def valueCount(data):
    value_counts = defaultdict(int)
    for d in data:
        if d['value'] != None:
            string = "".join([x for x in d['value'] if x not in ['$',',']])
            val = float(string)
            if val <= 1000 and val%100==0:
                value_counts[val] += 1
    value_counts = [(value_counts[k], k) for k in value_counts]
    value_counts.sort()
    value_counts.reverse()
    #print "Number of categories: {}\nTop 10 categories: {}".format(len(category_counts), category_counts[:10])
    #print "Mean number of items per categories: {}\nStd dev of items per categories: {}".format(np.mean([x[0] for x in category_counts]), np.std([x[0] for x in category_counts]))
    return value_counts

word_counts = wordCount(data)
category_counts = categoryCount(data)
value_counts = valueCount(data)
word_id = {x[1]: num for (x, num) in zip(word_counts, range(len(word_counts)))}
category_id = {x[1]: num for (x, num) in zip(category_counts, range(len(category_counts)))}
value_id = {x[1]: num for (x, num) in zip(value_counts, range(len(value_counts)))}

# for key in value_counts:
#     print("ValCounts, Val: " + str(key))


#word frequency per category for kategory prediction
wordFreq = defaultdict(float)
wordFreqCat = {}
for k in (category_id.keys()):
   wordFreqCat[category_id[k]] = defaultdict(float)
punctuation = string.punctuation

#category word count totals
for d in data:
    if d['category'] in category_id.keys():
        words = ''.join([c for c in d['question'].lower() if not c in punctuation]).split()
        for w in words: # and if w not in stopwords:
            wordFreq[w] += 1.0 # Word frequency
            wordFreqCat[category_id[d['category']]][w] += 1.0 # Per-category frequency

totalWords = sum(wordFreq.values())
for w in wordFreq:
  wordFreq[w] /= totalWords

for c in range(len(category_id)):
  totalWordsCat = sum(wordFreqCat[c].values())
  for w in wordFreqCat[c]:
    wordFreqCat[c][w] /= totalWordsCat

for c in range(len(category_id)):
  diffs = []
  for w in wordFreq:
    diffs.append((wordFreqCat[c][w] - wordFreq[w], w))
  diffs.sort()
  diffs.reverse()
  #print("10 words with maximum frequency difference for category " + str(c) + " = " + str(diffs[:10]))

topWords = [(wordFreq[w], w) for w in wordFreq]
topWords.sort()
topWords.reverse()
print("len top words: " + str(len(topWords)))
commonWords = [x[1] for x in topWords[:1000]]
commonWordsPositions = dict(zip(commonWords, range(len(commonWords))))
commonWordsSet = set(commonWords)



def feature(r):
  feat = [0] * len(commonWords)
  rText = ''.join([c for c in (r['question'].lower() + r['answer'].lower()) if not c in punctuation])
  for w in rText.split():
    if w in commonWordsSet:
      feat[commonWordsPositions[w]] = 1
  return feat



print ("word count: " + str(len(wordCount(data))))
print ("value count: " + str(len(valueCount(data))))
print("data length: " + str(len(data)))
sys.stdout.flush()

#train_raw_data = data[:(len(data)/2)]
#val_raw_data = data[(len(data)/2):]

X = []
y = []
for d in data:
    if d['value'] != None:
        string = "".join([x for x in d['value'] if x not in ['$',',']])
        val = float(string)
        if val <= 1000 and val%100==0:
            x = [0] * len(word_id)
            words = ''.join([c for c in d['question'].lower() if not c in punctuation]).split()
            for w in words:
                if w in word_id.keys():
                    x[word_id[w]] = 1
            X.append(x)
            if val in value_id.keys():
                y.append(value_id[val])
            else:
                print("wasnt in value_id.keys(): " + str(val))

print("Done loading data!")
print("length X: " + str(len(X)))
sys.stdout.flush()

#these train and validate for value prediction based on word text
X_train, X_val = np.asarray(X[:(len(X)/2)]), np.asarray(X[(len(X)/2):])
y_train, y_val = np.asarray(y[:(len(y)/2)]), np.asarray(y[(len(y)/2):])

#these train and validate for category prediction based on word and answer text
X_train2 = [feature(r) for r in data[:len(data)/2] if r['category'] in category_id.keys()]
y_train2 = [category_id[r['category']] for r in data[:len(data)/2] if r['category'] in category_id.keys()]
X_val2 = [feature(r) for r in data[len(data)/2:] if r['category'] in category_id.keys()]
y_val2 = [category_id[r['category']] for r in data[len(data)/2:]  if r['category'] in category_id.keys()]

print("done splitting models")
print("length X2train: " + str(len(y_train2)))
print("length X2val: " + str(len(y_val2)))
sys.stdout.flush()

model = LinearSVC(C=1)
model.fit(X_train, y_train)
# predictions = [int(x) for x in model.predict(X_val)]
# acc = [(x == y) for (x,y) in zip(predictions, y_val)]
# acc = sum(acc) * 1.0 / len(acc)

print "Score: {}".format(model.score(X_val, y_val))

sys.stdout.flush()

model2 = LinearSVC(C=1)
model2.fit(X_train2, y_train2)
# predictions2 = [int(x) for x in model.predict(X_val2)]
# acc2 = [(x == y) for (x,y) in zip(predictions2, y_val2)]
# acc2 = sum(acc2) * 1.0 / len(acc2)

print "Score2: {}".format(model.score(X_val2, y_val2))

sys.stdout.flush()
