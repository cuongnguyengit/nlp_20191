import collections
import numpy as np
import time
import configparser
import json
import os
import shutil
import datetime
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import tensorflow as tf
    from tensorflow.contrib import rnn

class trainerRNN:
    def __init__(self):
        # load config
        Config = configparser.ConfigParser()
        Config.read("setting.ini")
        self.window = int(Config.get('RNN', 'window'))
        self.time_steps = self.window - 1
        self.num_hidden = int(Config.get('RNN', 'num_hidden'))
        self.num_input = int(Config.get('RNN', 'num_input'))
        self.batch_size = int(Config.get('RNN', 'batch_size'))
        self.iteration = int(Config.get('RNN', 'iteration'))
        self.data_path = str(Config.get('RNN', 'data_path'))
        self.model_path = str(Config.get('RNN', 'model_path'))

    def plot_Mertric(self, np_data1, np_data2):
        plt.subplot(211)
        plt.plot(np_data1)
        plt.ylabel('Loss')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(np_data2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('TrainingMertric.png')

    def chunks(self, l, n, truncate=False):
        """Yield successive n-sized chunks from l."""
        batches = []
        for i in range(0, len(l), n):
            if truncate and len(l[i:i + n]) < n:
                continue
            batches.append(l[i:i + n])
        return batches

    def chunks_list_line(self, list_lines, truncate=False):
        batches = []
        for line in list_lines:
            batches.extend(self.chunks(line.strip().split(), self.window, truncate))
        return batches

    def word_indexing(self, words):
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

    def data_sampling(self, list_lines):
        t = time.time()
        content = ' '.join(list_lines)
        words = content.split()
        vocab_dict, rev_vocab_dict = self.word_indexing(words)
        with open('vocab/rev_vocab.json', 'w', encoding='utf-8') as fp:
            json.dump(rev_vocab_dict, fp)
        with open('vocab/vocab.json', 'w', encoding='utf-8') as fp:
            json.dump(vocab_dict, fp)
        training_data = []
        samples = self.chunks_list_line(list_lines, truncate=True)
        for sample in samples:
            training_data.append(([vocab_dict[z] for z in sample[:-1]], vocab_dict[sample[-1:][0]]))
        print('Total samples: ', len(samples))
        print('data_sampling', time.time() - t)
        return training_data, len(set(words))

    def RNN(self, x, weights, biases):

        # Unstack to get a list of 'timesteps' tensors, each tensor has shape (batch_size, n_input)
        x = tf.unstack(x, self.time_steps, 1)

        # Build a LSTM cell
        lstm_cell = rnn.BasicLSTMCell(self.num_hidden)

        # Get LSTM cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def train_model(self):
        f = open(self.data_path, 'r', encoding='utf-8')
        list_lines = list(f.readlines())
        f.close()
        training_data, num_classes = self.data_sampling(list_lines)
        print('Class: ', num_classes)
        training_data, testing_data = train_test_split(training_data, test_size=0.1, shuffle=True)
        # Build the Batches:
        batches = self.chunks(training_data, self.batch_size)
        # print(batches)

        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # tf graph input
        X = tf.placeholder("float", [None, self.time_steps, self.num_input], name='X')
        Y = tf.placeholder("float", [None, num_classes])

        print('Init variable')

        logits = self.RNN(X, weights, biases)

        y_pred_softmax = tf.nn.softmax(logits, name='y_pred5')
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')
        y_true = tf.argmax(Y, 1)

        # Loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_op)
        correct_pred = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables with default values
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        total_time_train = 0
        print("Training")

        Training_Accuracy = []
        Loss = []

        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            print('INIT')
            for i in range(0, self.iteration):
                t = time.time()
                loss_list = []
                acc_list = []
                for j, batch in enumerate(batches):
                    # print(j)
                    X_batch = [x[0] for x in batch]
                    Y_batch = [x[1] for x in batch]
                    Y_batch_encoded = []
                    for x in Y_batch:
                        one_hot_vector = np.zeros([num_classes], dtype=float)
                        one_hot_vector[x] = 1.0
                        Y_batch_encoded.append(one_hot_vector)
                    Y_batch_encoded = np.vstack(Y_batch_encoded)
                    X_batch = np.vstack(X_batch)
                    X_batch = X_batch.reshape(len(batch), self.time_steps, self.num_input)
                    Y_batch_encoded = Y_batch_encoded.reshape(len(batch), num_classes)
                    _, acc, loss, onehot_pred = sess.run(
                        [train_op, accuracy, loss_op, logits], feed_dict={X: X_batch, Y: Y_batch_encoded})
                    loss_list.append(loss)
                    acc_list.append(acc)
                loss = sum(loss_list)/len(loss_list)
                acc = sum(acc_list)/len(acc_list)
                total_time_train += time.time() - t
                datetime_object = datetime.datetime.now()
                Training_Accuracy.append(float(acc * 100))
                Loss.append(float(loss))

                print(str(datetime_object) + ":_Iteration " + str(i + 1) + '/' + str(self.iteration) + ", Loss= " + "{:.4f}".format(loss)
                      + ", Training Accuracy= " + "{:.2f}".format(acc * 100)
                      + ', Time: ' + '{:.2f}'.format((time.time() - t) / 60))
                print('Total Time Spend: %.2f' % (total_time_train / 60))
            self.plot_Mertric(Loss, Training_Accuracy)
            inputs = {
                "X": X,
            }
            outputs = {"y_pred": y_pred}
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)
            tf.saved_model.simple_save(
                sess, self.model_path, inputs, outputs
            )

            # Testing
            X_test = [x[0] for x in testing_data]
            Y_test = np.array([x[1] for x in testing_data])
            X_test = np.vstack(X_test)
            X_test = X_test.reshape(len(testing_data), self.time_steps, self.num_input)
            Y_pred = sess.run(y_pred, feed_dict={X: X_test})
            # print(Y_pred)
            # print(Y_test)
            Correct_Test = list(np.equal(Y_pred, Y_test))
            Testing_Accuracy = 1.0 * Correct_Test.count(True) / len(Correct_Test)
            print('Testing Accuracy= ' + "{:.2f}".format(Testing_Accuracy * 100))


if __name__ == '__main__':
    trainer = trainerRNN()
    trainer.train_model()
