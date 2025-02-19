import os
import argparse
from utils import get_logger, load_vocab, get_processing_word, \
    get_trimmed_wordvec_vectors, get_random_wordvec_vectors, positional_embedding

class Config():
    def __init__(self, parser=None, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        ## parse args
        parser = argparse.ArgumentParser('CLASSIFIER argument for training')
        # training parameters
        self.parser = parser
        parser.add_argument('--gpu', default='1', type=str,
                            help='gpus')

        parser.add_argument('--epochs', default='50', type=int,
                    help='number of epochs')
        parser.add_argument('--dropout', default='0.5', type=float,
                    help='dropout')
        parser.add_argument('--batch_size', default='2', type=int,
                    help='batch size')

        #888888888888888888888
        parser.add_argument('--lr', default='0.0001', type=float,
                    help='learning rate')
        parser.add_argument('--lr_method', default='RMSprop', type=str,
                    help='optimization method')
        parser.add_argument('--min_lr', default=0.00001, type=float,
                            help='min learning')

        parser.add_argument('--lr_decay', default='0.00001', type=float,
                    help='learning rate decay rate')
        parser.add_argument('--momentum', default='0.99', type=float,
                            help='momentum')
        parser.add_argument('--gamma', default='0.99', type=float,
                           help='gamma')
        parser.add_argument('--step_size', default='1', type=int,
                            help='step_size')

        #********************
        parser.add_argument('--clip', default='10', type=float,
                    help='gradient clipping')
        parser.add_argument('--l2_reg_lambda', default='0.001', type=float,
                    help='l2 regularization coefficient')
        parser.add_argument('--nepoch_no_imprv', default='6', type=int,
                            help='number of epoch patience')
        parser.add_argument('--num_layer', default='1', type=int,
                            help='number of layer')
        # data and results paths
        parser.add_argument('--dir_output', default='output/', type=str,
                    help='directory for output')
        parser.add_argument('--dir_out_class', default='output/', type=str,
                            help='directory for output')
        parser.add_argument('--data_root', default='data_fold/', type=str,
                            help='directory for dataset')

        parser.add_argument('--data_name', default='moxifloxacin_2', type=str,
                            help='name for dataset')
        parser.add_argument('--folds', default='split_9', type=str,
                            help='name for dataset')


        # model hyperparameters
        parser.add_argument('--attention_size', default='300', type=int,
                            help='attention_size')
        parser.add_argument('--num_units', default='300', type=int,
                            help='the dim of the query,key,value')
        parser.add_argument('--hidden_size', default='128', type=int,
                            help='the dim of the query,key,value')
        parser.add_argument('--hidden_size_lstm_sentence', default='150', type=int,
                            help='hidden size of sentence level lstm')
        parser.add_argument('--hidden_size_lstm_document', default='150', type=int,
                            help='hidden size of document level lstm')


        self.parser.parse_args(namespace=self)

        self.dir_model  = os.path.join(self.dir_output + '/' + self.data_name + '/' + self.folds)
        self.path_log   = os.path.join(self.dir_output + '/' + self.data_name + '/' + self.folds, "log_attention"
                                                                                                  ".txt")

        # dataset
        self.filename_dev = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'dev.txt')
        self.filename_test = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'test.txt')
        self.filename_train = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'train.txt')
        # filename_train = os.path.join(args.data_root + args.data_name + '/' + args.folds, 'shuju.txt')

        self.word_position_dev = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'dev_wordposition.txt')
        self.word_position_test = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'test_wordposition.txt')
        self.word_position_train = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'train_wordposition.txt')
        # word_position_train = os.path.join(args.data_root + args.data_name + '/' + args.folds,'supconword.txt')

        self.sentence_position_dev = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'dev_sentenceposition.txt')
        self.sentence_position_test = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'test_sentenceposition.txt')
        self.sentence_position_train = os.path.join(self.data_root + self.data_name + '/' + self.folds, 'train_sentenceposition.txt')
        # sentence_position_train = os.path.join(args.data_root + args.data_name + '/' + args.folds,'supconsentence.txt')

        # vocab
        self.filename_words = os.path.join('data/words.txt')
        self.filename_tags = os.path.join('data/tags.txt')

        # directory for training outputs
        if not os.path.exists('data'):
            os.makedirs('data')

        # directory for data output
        if not os.path.exists(self.dir_output+ '/' + self.data_name+ '/' + self.folds):
            os.makedirs(self.dir_output+ '/' + self.data_name+ '/' + self.folds)
        if not os.path.exists(self.dir_out_class):
            os.makedirs(self.dir_out_class)

        # create instance of logger
        logger = get_logger(self.path_log)

        # log the attributes
        msg = ', '.join(['{}: {}'.format(attr, getattr(self, attr)) for attr in dir(self) \
                        if not callable(getattr(self, attr)) and not attr.startswith("__")])
        logger.info(msg)


        # load if requested (default)
        if load:
            self.load()


    def load(self):
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)

        # 3. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)
