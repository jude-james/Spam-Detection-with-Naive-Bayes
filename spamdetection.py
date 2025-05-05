import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.classify.api import ClassifierI
import random
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# TODO uncomment these in final submission

class ConfusionMatrix:
    def __init__(self,predictions,goldstandard,classes=(1,0)):
        
        (self.c1,self.c2) = classes
        #self.predictions=predictions
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        for p,g in zip(predictions, goldstandard):
            if g == self.c1:
                if p == self.c1:
                    self.TP += 1
                else:
                    self.FN += 1
            elif p == self.c1:
                self.FP += 1
            else:
                self.TN += 1

    def precision(self):
        p = self.TP / (self.TP + self.FP)
        return p

    def recall(self):
        r = self.TP / (self.TP + self.FN)
        return r

    def f1(self):
        p=self.precision()
        r=self.recall()
        f1=2*p*r/(p+r)
        return f1

class SimpleClassifier_mf(ClassifierI):
    def __init__(self, k):
        self._k = k
        self._pos = set()
        self._neg = set()

    def _most_frequent_words(self, posfreq, negfreq):
        difference = posfreq - negfreq
        sorteddiff = difference.most_common()
        justwords = [word for (word, freq) in sorteddiff[:self._k]]
        return set(justwords)

    def train(self, training_data):
        pos_freq_dist = FreqDist()
        neg_freq_dist = FreqDist()

        for features, label in training_data:
            if label == 1:
                pos_freq_dist += features
            else:
                neg_freq_dist += features

        self._pos = self._most_frequent_words(pos_freq_dist, neg_freq_dist)
        self._neg = self._most_frequent_words(neg_freq_dist, pos_freq_dist)

    def classify(self, doc):
        score = 0
        for word, value in doc.items():
            if word in self._pos:
                score += value
            if word in self._neg:
                score -= value
        return 0 if score < 0 else 1

    def labels(self):
        return (1, 0)

# Load the data
train_data = pd.read_csv('spam_detection_training_data.csv')
test_data = pd.read_csv('spam_detection_test_data.csv')

train_text = train_data['text'].values
train_label = train_data['label'].values

test_text = test_data['text'].values

# Preprocess a word list, convert to lowercase, lemmetize, remove stopwords and punctuation
stop = stopwords.words('english')
lemm = WordNetLemmatizer()

def normalise(wordlist):
    lowered = [word.lower() for word in wordlist]
    lemmatized = [lemm.lemmatize(word) for word in lowered]
    filtered = [word for word in lemmatized if word.isalpha() and word not in stop]
    return filtered

def vocabulary_size(sentences):
    tok_counts = {}
    for sentence in sentences:
        for token in sentence:
            tok_counts[token]=tok_counts.get(token,0)+1
    return len(tok_counts.keys())

# Compare size of vocabulary before and after preprocessing
raw_vocab_size = vocabulary_size([word_tokenize(text) for text in train_text])
normalised_vocab_size = vocabulary_size([normalise(word_tokenize(text)) for text in train_text])

print("Raw vocab size:", raw_vocab_size)
print("Normalised vocab size:", normalised_vocab_size)
print("Normalisation produced a {0:.2f}% reduction in vocabulary size from {1} to {2}".format(
    100*(raw_vocab_size - normalised_vocab_size)/raw_vocab_size, raw_vocab_size, normalised_vocab_size))

# Split the training data into training and testing data, so we can evaluate the test data with its labels
def split_data(data, ratio=0.7): # when the second argument is not given, it defaults to 0.7
    """
    Given collection of items and ratio:
     - partitions the collection into training and testing, where the proportion in training is ratio,

    :param data: A list (or generator) of documents or doc ids
    :param ratio: The proportion of training documents (default 0.7)
    :return: a pair (tuple) of lists where the first element of the
            pair is a list of the training data and the second is a list of the test data.
    """

    n = len(data)
    train_indices = random.sample(range(n), int(n * ratio))    
    test_indices = list(set(range(n)) - set(train_indices))

    train = [data[i] for i in train_indices]      
    test = [data[i] for i in test_indices]

    return (train, test)

def get_train_test_split():
    # Split the training data into spam and not spam, so the ratio of train and test is the same
    train_data_full = list(zip(train_text, train_label))
    train_data_pos = [data for data in train_data_full if data[1] == 1]
    train_data_neg = [data for data in train_data_full if data[1] == 0]

    # Use a random seed to get consistant results, then split into train and test with 0.8 ratio
    random.seed(67)
    train_data_pos_split, test_data_pos_split = split_data(train_data_pos, 0.8)
    train_data_neg_split, test_data_neg_split = split_data(train_data_neg, 0.8)

    # Combine the positive and negative splits
    train_data_split = train_data_pos_split + train_data_neg_split
    test_data_split = test_data_pos_split + test_data_neg_split

    # Convert training data split into a frequency distribuation
    train_data_split_freq_dist = [(FreqDist(normalise(word_tokenize(text))), label) for text, label in train_data_split]

    # Convert test data split into a frequency distribution (this time there are labels because its from the training data)
    test_data_split_freq_dist = [(FreqDist(normalise(word_tokenize(text))), label) for text, label in test_data_split]

    return train_data_split_freq_dist, test_data_split_freq_dist

train_data_split_freq_dist, test_data_split_freq_dist = get_train_test_split()

# Get the documents to train and the correct labels to evaluate
docs, goldstandard = zip(*test_data_split_freq_dist)

# Use a word list based classifiction technique for evaluation
simple_classifier = SimpleClassifier_mf(100)
simple_classifier.train(train_data_split_freq_dist)
cm = ConfusionMatrix(simple_classifier.classify_many(docs), goldstandard)

print("word list classifier precision:", cm.precision())
print("word list classifier recall:", cm.recall())
print("word list classifier f1:", cm.f1())

# Use a Naive Bayes classification technique with NLTK for evaluation
naive_bayes_classifier = NaiveBayesClassifier.train(train_data_split_freq_dist)
cm = ConfusionMatrix(naive_bayes_classifier.classify_many(docs), goldstandard)

print("naive bayes precision:", cm.precision())
print("naive bayes recall:", cm.recall())
print("naives bayes f1:", cm.f1())

# Convert the training data into a frequency distribuation (and tokenise + normalise the text)
train_data_freq_dist = [(FreqDist(normalise(word_tokenize(text))), label) for text, label in zip(train_text, train_label)]

# Convert testing data into a frequency distribution (no labels given for testing)
test_data_freq_dist = [FreqDist(normalise(word_tokenize(text))) for text in test_text]

# Use a Naive Bayes classification technique with NLTK for final predictions on all test data
naive_bayes_classifier = NaiveBayesClassifier.train(train_data_freq_dist)

# all docs
predictions = naive_bayes_classifier.classify_many(doc for doc in test_data_freq_dist)
predictions = [int(pred) for pred in predictions] # converting from np.int64() to int
print(predictions)

# a single doc (custom string example) REMOVE this later, probably not needed
prediction = naive_bayes_classifier.classify(FreqDist(normalise(word_tokenize("thank you for your email."))))
print(prediction)

naive_bayes_classifier.show_most_informative_features(20)
# TODO graph or table this in report

# Save prediction labels to a csv file
def save_as_csv(pred_labels, location = '.'):
    """
    Save the labels out as a .csv file
    :pred_labels: numpy array of shape (no_test_labels,) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert pred_labels.shape[0]==1552, 'wrong number of labels, should be 1552 test labels'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',')

save_as_csv(np.array(predictions)) # TODO turn into clean ints