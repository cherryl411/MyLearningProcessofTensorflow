# -*- coding: utf-8 -*-

"""

@project: tensorflow_cookbook
@author: Cherry_L411@163.com
@file: SkipGram.py
@date: 2018-12-28

"""
import Data.DataDeal as dd
import os
from urllib.request import urlopen
import io
import tarfile
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import tensorflow as tf

fatherPath = os.path.abspath('.')
graderFatherPath = os.path.abspath('..')
# print(graderFatherPath)

# English Stop Words
stopword = stopwords.words('english')

# load two classfier data from the direct url
def load_two_classifier_data(saveFloderName, saveName, url = None):
    stream_data = urlopen(url)
    tmp = io.BytesIO()
    while 1:
        s = stream_data.read()
        if not s:
            break
        tmp.write(s)
        stream_data.close()
        tmp.seek(0)
    tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
    pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
    neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')

    # save pos/neg reviews
    pos_data1 = []
    neg_data1 = []
    for line in pos:
        pos_data1.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
    for line in neg:
        neg_data1.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
    tar_file.close()

    # write to file
    data_file_path = saveFloderName + saveName

    if not os.path.exists(saveFloderName):
        os.makedirs(saveFloderName)
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)
    else:
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)

    pos_file = data_file_path + '/rt-polarity.pos'
    neg_file = data_file_path + '/rt-polarity.neg'

    # save files
    with open(pos_file, 'w') as p:
        p.write(''.join(pos_data1))
    p.close()

    with open(neg_file, 'w') as p:
        p.write(''.join(neg_data1))
    p.close()

    return pos_data1, neg_data1

# Build the dictionary, return pairs of word and its id
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    sentences_split = [x.split() for x in sentences]
    words = [x for sublist in sentences_split for x in sublist]

    # Initialize list of [word, word_count] for each words, starting with unknown
    count = [('RARE', -1)]

    # Add most frequent words into count
    count.extend(Counter(words).most_common(vocabulary_size))

    # create dictionary
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    return word_dict

# Build sentences to lists of words'id
def text_to_id(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sen in sentences:
        sen_data = []
        for word in sen:
            if word in word_dict:
                word_id = word_dict[word]
            else:
                word_id = 0
            sen_data.append(word_id)
        data.append(sen_data)

    return (data)

# Generate the train data which the type is similar with (input, output), the input is target word, the output are context words.
def generate_batch_data(sentences, batch_size, windows_size, method='skip-gram'):
    # Fill up data batch
    batch_data = []
    labels_data = []

    while len(batch_data) < batch_size:
        random_data = np.random.choice(sentences)

        # Create consecute windows to look at
        windows_sequence = [random_data[max(ix-windows_size, 0):(ix+windows_size+1)] for ix, x in enumerate(random_data)]

        # Denote center word
        label_index = [ix if ix < windows_size else windows_size for ix, x in enumerate(random_data)]

        # Pull out center word and its output words as a turple
        if method == 'skip-gram':
            batch_labels = [(x[y], x[:y]+x[y+1:]) for x, y in zip(windows_sequence, label_index)]
            # Change it to a list of turple like (target_word, context_word)
            turple_data = [(x, y1) for x, y in batch_labels for y1 in y]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        batch, label = [list(x) for x in zip(*turple_data)]
        batch_data.extend(batch[:batch_size])
        labels_data.extend((label[:batch_size]))

    batch_data = batch_data[:batch_size]
    labels_data = labels_data[:batch_size]

    # Cover to numpay array
    batch_data = np.array(batch_data)
    labels_data = np.transpose(np.array([labels_data]))

    return (batch_data, labels_data)



saveFloderName = graderFatherPath + '/Data'
saveName = '/rt-polaritydata'
pos_data, neg_data = dd.judge_data(saveFloderName, saveName)
if not pos_data or not neg_data:
    print('Load Data From URL!')
    pos_data, neg_data = load_two_classifier_data(saveFloderName=saveFloderName, saveName=saveName, url='http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz')

print('Data Ready!')

text = pos_data + neg_data
target = [1] * len(pos_data) + [0]*len(neg_data)

text_n = dd.text_normalized(text, stopword)

vocabulary_size = 10000
word_dictionary = build_dictionary(text_n, vocabulary_size)  # {word, id}
word_dictionary_reverse = dict(zip(word_dictionary.values(), word_dictionary.keys()))  # {id, word}
text_data = text_to_id(text_n, word_dictionary)

# xx = generate_batch_data(text_data, 50, 2, 'skip-gram')
# Embedding matrix and placeholder declaration
embedding_size = 200
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Inputs and target declaration
batch_size = 50
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
valid_examples = [word_dictionary[x] for x in valid_words]
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# NCE(noise-contrastive error)
nce_weight = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
num_sampled = int(batch_size/2)
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_biases, inputs=embed, labels=y_target, num_sampled=num_sampled, num_classes=vocabulary_size))

# Similarity calculate
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embedding = embeddings / norm
valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
similarity = tf.matmul(valid_embedding, normalized_embedding, transpose_b=True)

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
loss_vec = []
loss_x_vec = []
generation_size = 10000
for i in range(generation_size):
    batch_input, batch_label = generate_batch_data(text_data, batch_size, 2, 'skip-gram')
    feed_dict = {x_inputs: batch_input, y_target: batch_label}
    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
    # Print the loss
    print_loss_time = 200
    if (i+1)%print_loss_time == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at setp {} is {}'.format(i+1, loss_val))

    # Print the most related words of valid_words
    print_valid_time = 2000
    if (i+1)%print_valid_time == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_reverse[valid_examples[j]]
            topN = 3
            nearest = (-sim[j, :]).argsort()[1:topN+1]
            log_str = 'Nearest to {}:'.format(valid_word)
            for k in range(topN):
                closed_word = word_dictionary_reverse[nearest[k]]
                if k == topN-1:
                    log_str = '{} {}'.format(log_str, closed_word)
                else:
                    log_str = '{} {},' .format(log_str, closed_word)
            print(log_str)