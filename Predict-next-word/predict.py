import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
import json
import os
import re
import unicodedata as ud


class PredictModel:
    def __init__(self, model='model', path_vocab='vocab/vocab.json', path_rev_vocab='vocab/rev_vocab.json'):
        with open(path_vocab, encoding='utf-8') as handle:
            self.vocab = json.loads(handle.read())
        with open(path_rev_vocab, encoding='utf-8') as handle:
            self.rev_vocab = json.loads(handle.read())
        self.model = model

        model1 = model + '1/'
        model2 = model + '2/'
        model3 = model + '3/'
        model4 = model + '4/'
        model5 = model + '5/'

        self.X1, self.Y1, self.sess1 = self.load_model_tf(model1)
        self.X2, self.Y2, self.sess2 = self.load_model_tf(model2)
        self.X3, self.Y3, self.sess3 = self.load_model_tf(model3)
        self.X4, self.Y4, self.sess4 = self.load_model_tf(model4)
        self.X5, self.Y5, self.sess5 = self.load_model_tf(model5)
        self.sess = [self.sess1, self.sess2, self.sess3, self.sess4, self.sess5]
        self.X = [self.X1, self.X2, self.X3, self.X4, self.X5]
        self.Y = [self.Y1, self.Y2, self.Y3, self.Y4, self.Y5]

    def load_model_tf(self, modelx):
        if not os.path.isdir(modelx):
            return 0, 0, 0
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                modelx,
            )
            X = graph.get_tensor_by_name('X:0')
            y_pred = graph.get_tensor_by_name('y_pred:0')
            y_pred5 = graph.get_tensor_by_name('y_pred5:0')
            return X, y_pred5, sess

    def sylabelize(self, text):
        text = ud.normalize('NFC', text)

        specials = ["==>", "->", "\.\.\.", ">>"]
        digit = "\d+([\.,_]\d+)+"
        email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
        #web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
        web = "\w+://[^\s]+"
        #datetime = [
        #    "\d{1,2}\/\d{1,2}(\/\d{1,4})(^\dw. )+",
        #    "\d{1,2}-\d{1,2}(-\d+)?",
        #]
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]

        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(specials)
        patterns.extend([web, email])
        #patterns.extend(datetime)
        patterns.extend([digit, non_word, word])

        patterns = "(" + "|".join(patterns) + ")"
        tokens = re.findall(patterns, text, re.UNICODE)

        return ' '.join([token[0] for token in tokens])

    def process(self, text):
        list_token = self.sylabelize(text).split()
        decode_list = []
        for token in list_token:
            if token in self.vocab:
                decode_list.append(self.vocab[token])
        return decode_list

    def predict_list(self, text):
        batches = self.process(text)
        dict_result = {1: [], 2: [], 3: [], 4: [], 5: []}
        for i in range(0, 5):
            if len(batches) >= i + 1:
                batch = []
                batch.extend(batches[-i-1:])
                batch = np.array(batch).reshape(1, i + 1, 1)
                result = self.sess[i].run(self.Y[i], feed_dict={self.X[i]: batch})
                temp = np.argsort(result[0])[-3:]
                kq = []
                for j in temp:
                    kq.append((str(self.rev_vocab[str(j)]), 1.0 * int(result[0][j] * 100) / 100))
                dict_result[i+1].extend(kq)
        print(text)
        print(dict_result)
        return dict_result

if __name__ == '__main__':
    preLSTM = PredictModel()
    preLSTM.predict_list('làm việc với các bạn tuy khá vất')


# 5 words as input
# text = "truyền hình và các tạp chí ẩm"
# words = text.split()
# print(words)
# with open('vocab/vocab.json') as handle:
#     vocab = json.loads(handle.read())
# with open('vocab/rev_vocab.json') as handle:
#     rev_vocab = json.loads(handle.read())
#
# input = []
# for word in words:
#     if word in vocab:
#         print(word)
#         input.append(vocab[word])
#     else:
#         input.append(int(-1))
# print(input)
# X_batch = np.array(input)
# X_batch = X_batch[-5:].reshape(1, 5, 1)
# graph = tf.Graph()
# with graph.as_default():
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(
#             sess,
#             [tag_constants.SERVING],
#             'model/',
#         )
#         X = graph.get_tensor_by_name('X:0')
#         y_pred = graph.get_tensor_by_name('y_pred:0')
#         y_pred5 = graph.get_tensor_by_name('y_pred5:0')
#
#         result = sess.run(y_pred, feed_dict={X: X_batch})
#         result5 = sess.run(y_pred5, feed_dict={X: X_batch})
#
#         print(result)
#         temp = np.argsort(result5[0])[-5:]
#         kq = []
#         for i in temp:
#             kq.append(str(rev_vocab[str(i)]))
#         print(kq)
#         print("the next word is '{}'".format(str(rev_vocab[str(result[0])])))
