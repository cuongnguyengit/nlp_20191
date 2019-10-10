import collections
from pyvi import ViTokenizer
import time
import json


def chunks(l, n, truncate=False):
    """Yield successive n-sized chunks from l."""
    t = time.time()
    batches = []
    for i in range(0, len(l), n):
        if truncate and len(l[i:i + n]) < n:
            continue
        batches.append(l[i:i + n])
    print('chunks', time.time() - t)
    return batches


def word_indexing(words):
    t = time.time()
    """

    :param words: a string
    :return: a vocabulary dictionary {word1: 1, word2: 2,  ...} and
     its reveres {1: word1, 2: word2, ...}
    """
    vocab = collections.Counter(words).most_common()
    vocab_dict = dict()
    for word, _ in vocab:
        vocab_dict[word] = len(vocab_dict)
    rev_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))

    print('word_indexing', time.time() - t)
    return vocab_dict, rev_vocab_dict


def data_sampling(content, window):
    t = time.time()
    """

    :param content: Text vocab as string
    :param window: Window size for sampling, the window moves on the text vocab to build the samples
    :return: Training vocab includes (input, label) pair and number of classes

    If the window includes "cats like to chase mice" X is "cats like to chase" and y is "mice"
    """
    # words = nltk.tokenize.word_tokenize(content)
    print('tokenize')
    # words = ViTokenizer.tokenize(content).split()
    words = content.split()
    print('word_indexing')
    vocab_dict, rev_vocab_dict = word_indexing(words)

    with open('vocab/rev_vocab.json', 'w', encoding='utf-8') as fp:
        json.dump(rev_vocab_dict, fp)
    with open('vocab/vocab.json', 'w', encoding='utf-8') as fp:
        json.dump(vocab_dict, fp)

    training_data = []
    # samples = chunks(words, window, truncate=True)
    # for index, sample in enumerate(samples):
    #     print(index, sample)
    #     training_data.append(([vocab_dict[z] for z in sample[:-1]], vocab_dict[sample[-1:][0]]))

    print('data_sampling', time.time() - t)
    # print(training_data)
    print(len(vocab_dict))
    return training_data, len(set(words))


with open("Data/TrainingData.txt", 'r', encoding='utf-8') as f:
    content = f.read()

window = 6
time_steps = window - 1
num_hidden = 512
num_input = 1
batch_size = 150
iteration = 250

training_data, num_classes = data_sampling(content, window=window)
print(num_classes)

# Build the Batches:
# batches = chunks(training_data, batch_size)
# print(batches)