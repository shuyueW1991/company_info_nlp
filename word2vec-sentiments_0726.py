# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
                    #yield TaggedDocument(line.split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
             #       self.sentences.append(TaggedDocument(line.split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)


log.info('source load')
#sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
#sources = {'all20.txt':'TEST_NEG', 'all22.txt':'TEST_POS', 'all23.txt':'TRAIN_NEG', 'all24.txt':'TRAIN_POS', 'all16.txt':'TRAIN_UNS'}
#sources = {'test_neg.txt1':'TEST_NEG', 'test_pos.txt1':'TEST_POS', 'train_neg.txt1':'TRAIN_NEG', 'train_pos.txt1':'TRAIN_POS', 'train_unsup.txt1':'TRAIN_UNS'}
sources = {'test_neg.txt2':'TEST_NEG', 'test_pos.txt2':'TEST_POS', 'train_neg.txt2':'TRAIN_NEG', 'train_pos.txt2':'TRAIN_POS', 'train_unsup.txt2':'TRAIN_UNS'}

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
#model = Doc2Vec(min_count=1, window=10, size=1000, sample=1e-4, negative=5, workers=7)

model = Doc2Vec(min_count=1, window=100, size=1500, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(1):
	log.info('EPOCH: {}'.format(epoch))
#	model.train(sentences.sentences_perm())
	model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=model.iter)

log.info('Model Save')
#model.save('./imdb0726.d2v')
model.save('./imdb0726_1.d2v')
#model = Doc2Vec.load('./imdb0726.d2v')
model = Doc2Vec.load('./imdb0726_1.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((360, 2000))
train_labels = numpy.zeros(360)

for i in range(180):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[180 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[180 + i] = 0

log.info(train_labels)

test_arrays = numpy.zeros((140, 2000))
test_labels = numpy.zeros(140)

for i in range(70):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[70 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[70 + i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

log.info(classifier.score(test_arrays, test_labels))
