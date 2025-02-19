
"""
数据处理
"""
import sys
import random
import pandas as pd
import numpy as np
import logging
import time

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
WORD_PAD = '$W_PAD$'
TAG_PAD = '$T_PAD$'


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)

class Dataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename,
                 processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
    def __iter__(self):
        niter = 0

        tags = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            sentences = []
            # print("utili:",self.filename)
            for line in f:
                # print(len(sentences))
                if not line:
                    if len(sentences) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield sentences, tags
                        sentences, tags = [], []
                else:
                    sentence = line.replace('\n', '').split(',')[2]
                    # print(sentence)

                    tag = line.replace('\n', '').split(',')[1]

                    templist = list(sentence.split('.'))[:-1]
                    # print("temp:",templist)
                    for temp in templist:
                        tl = list(temp.split())
                        # print('tl:',tl)
                        # print("processing_word:",self.processing_word)
                        if self.processing_word is not None:
                            sentence = [self.processing_word(word) for word in tl]
                        sentences += [sentence]
                   #sencetences 一个句子的词对应的词典的索引
                    # print("sentences",sentences)

                    # exit()
                    # print(sentences)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    # print(tag)
                    tags.append(tag)
                    # print(tags)
                    yield sentences, tags
                    sentences, tags = [], []


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger

def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        file_dict: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d

def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects
        dataset=[sentences,tags]

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for sentences, tags in dataset:
            for sent in sentences:
                vocab_words.update(sent)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for sents, _ in dataset:
        for sent in sents:
            for word in sent:
                vocab_char.update(word)

    return vocab_char

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word
    return f

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word
    return f

def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w",encoding='utf-8') as f:
        if isinstance(vocab, dict):
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\t{}\n".format(word, vocab[word]))
                else:
                    f.write('{}\t{}'.format(word, vocab[word]))
        else:
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)
    print("- done. {} tokens".format(len(vocab)))

def get_trimmed_wordvec_vectors(filename, vocab):
    print("trim")
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    f = open(filename, 'r',encoding='utf-8')
    next(f)
    dim = len(f.readline().strip().split()) - 1
    embeddings = np.random.uniform(-0.1, 0.1, size=(len(vocab)+1, dim))
    with open(filename, 'r',encoding='utf-8') as inFile:
        for line in inFile:
            line = line.strip().split()
            word = line[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array([float(item) for item in line[1:]])

    return embeddings

def get_random_wordvec_vectors(dim, vocab):
    """
        Args:
            dim: dim of random embedding
            vocab: vocab

        Returns:
            matrix of embeddings (np array)

    """
    # print('random')
    embedding_matrix = []
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    #每个词表的结果初始化
    # print(len(embedding_matrix))

    for i, char in enumerate(vocab):
        embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    # print(len(embedding_matrix))
    embedding_matrix.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    embedding_matrix = np.reshape(embedding_matrix, [-1, dim])
    embedding_matrix = embedding_matrix.astype(np.float32)
    # print(len(embedding_matrix))

    return embedding_matrix

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=2):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    # print("sequence:", sequences)
    if nlevels == 1:
        # Ensure elements are iterable
        sequences = [[x] if not isinstance(x, (list, tuple)) else x for x in sequences]
        max_length = max(map(lambda x: len(x), sequences))

        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        # print("sequence:", sequences)
        # Ensure elements and sub-elements are iterable
        sequences = [[[x] if not isinstance(x, (list, tuple)) else x for x in seq] for seq in sequences]
        max_length_sentence = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_sentence)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_document = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_sentence, max_length_document)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_document)

    elif nlevels == 3:
        # Ensure elements, sub-elements, and sub-sub-elements are iterable
        sequences = [[[[x] if not isinstance(x, (list, tuple)) else x for x in sen] for sen in seq] for seq in sequences]
        max_length_word = max([max([max(map(lambda x: len(x), sen)) for sen in seq]) for seq in sequences])
        max_length_sentence = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sentence_padded, sentence_length = [], []
            for sen in seq:
                # all words are same length now
                sp, sl = _pad_sequences(sen, pad_tok, max_length_word)
                sentence_padded += [sp]
                sentence_length += [sl]
            # all sentences are same length now
            sentence_padded, _ = _pad_sequences(sentence_padded, [pad_tok] * max_length_word, max_length_sentence)
            sentence_length, _ = _pad_sequences(sentence_length, 0, max_length_sentence)
            sequence_padded += [sentence_padded]
            sequence_length += [sentence_length]

        max_length_document = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [[pad_tok] * max_length_word] * max_length_sentence,
                                            max_length_document)
        sequence_length, _ = _pad_sequences(sequence_length, [0] * max_length_sentence, max_length_document)

    return sequence_padded, sequence_length



def minibatches(data, minibatch_size,
                train_word_position,train_sentence_position):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    word_batch, sentence_batch = [], []
    for i, (x, y) in enumerate(data):
        # print('data',x)
        # exit()

        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch,word_batch,sentence_batch
            x_batch, y_batch = [], []
            word_batch, sentence_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        word_batch += [train_word_position[i]]
        sentence_batch += [train_sentence_position[i]]

    if len(x_batch) != 0:
        yield x_batch, y_batch, word_batch, sentence_batch

import random

def minibatches_split(data, minibatch_size, train_word_position, train_sentence_position):
    # random.shuffle(data)
    positive_samples = []
    negative_samples = []
    x_batch, y_batch = [], []
    word_batch, sentence_batch = [], []
    # data = list(data)
    for i, (x, y) in enumerate(data):
        # print(y)
        if y == [0]:
            positive_samples.append((x, y, train_word_position[i], train_sentence_position[i]))
        else:
            negative_samples.append((x, y, train_word_position[i], train_sentence_position[i]))

    num_negative_samples = len(negative_samples)
    num_positive_samples = len(positive_samples)
    print("positive:",num_positive_samples,"negative:",num_negative_samples)
    half_minibatch_size = minibatch_size // 2

    random.shuffle(negative_samples)
    random.shuffle(positive_samples)

    start_index_neg = 0
    while start_index_neg < num_negative_samples:

        end_index_neg = min(start_index_neg + half_minibatch_size, num_negative_samples)

        for i in range(start_index_neg, end_index_neg):

            x, y, word_pos, sentence_pos = negative_samples[i]
            # print("negative:")
            # print("x", x)
            # print("y", y)
            x_batch.append(x)
            y_batch.append(y)
            word_batch.append(word_pos)
            sentence_batch.append(sentence_pos)

        remaining_neg_samples = half_minibatch_size - (end_index_neg - start_index_neg)
        # remaining_pos_samples = min(remaining_neg_samples, num_positive_samples)  # Ensure we don't exceed the positive samples
        # print("positive:", len(positive_samples))
        random.shuffle(positive_samples)  # Shuffle positive samples to ensure randomness
        random_positive_samples = positive_samples[: half_minibatch_size ]  # Select the first remaining_pos_samples samples
        # print(len(random_positive_samples))
        for sample in random_positive_samples:
            x, y, word_pos, sentence_pos = sample
            # print("positive:")
            #
            # print("x",x)
            # print("y",y)
            x_batch.append(x)
            y_batch.append(y)
            word_batch.append(word_pos)
            sentence_batch.append(sentence_pos)
        # print("x_batch", len(x_batch))
        # print("y_batch",len(y_batch))

        yield x_batch, y_batch, word_batch, sentence_batch
        x_batch, y_batch = [], []
        word_batch, sentence_batch = [], []

        start_index_neg = end_index_neg
def positional_encoding(file,position):
    # [[0,1,2,...,499],
    # [0,1,2,...,499],
    # ...
    # [0,1,2,...,499]]
    if position == 'word':
        wordposition = []
        with open(file, 'r', encoding='utf-8') as f:
            # print(file)
            for line in f:
                line = line.replace('\n', '')
                temps = []
                templist = list(line.split('.'))[:-1]
                for temp in templist:
                    t = []
                    if len(temp) == 1:
                        temp = int(temp)
                        t.append(temp)
                    else:
                        tl = list(temp.split())
                        tl = [int(i) for i in tl]
                        t.extend(tl)
                    temps.append(t)
                wordposition.append(temps)
                # print("word:",wordposition)
                # exit()
        return wordposition
    if position == 'sentence':
        sentenceposition = []
        with open(file, 'r', encoding='utf-8') as f:
            # print(file)
            for line in f:
                line = line.replace('\n', '')
                temps = []
                templist = list(line.split('.'))[:-1]
                for temp in templist:
                    # t = []
                    if len(temp) == 1:
                        temp = int(temp)
                        temps.append(temp)
                    else:
                        tl = list(temp.split())
                        tl = int(tl[0])
                        temps.append(tl)
                    # temps.append(t)
                sentenceposition.append(temps)
                # print(sentenceposition)
                # exit()
        return sentenceposition

def positional_embedding(sequence_max_length, position,dim):
    # a = []
    # for w in word_position:
    #     for i in w:
    #         for j in i:
    #             a.append(j)
    # sequence_max_length = np.array(a).max()
    po101 = np.zeros(dim)
    po101 = np.expand_dims(po101, 0)
    if position == 'word':
        position_embedding = np.array([[pos / np.power(10000, 2. * i / dim) for i in range(dim)]
                                 for pos in range(sequence_max_length)])
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])
        position_embedding = np.concatenate((po101,position_embedding))
        return position_embedding
    if position == 'sentence':
        position_embedding = np.array([[pos / np.power(10000, 2. * i / dim) for i in range(dim)]
                                       for pos in range(sequence_max_length)])
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])
        position_embedding = np.concatenate((po101, position_embedding))
        return position_embedding


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)







