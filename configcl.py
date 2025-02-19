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
        parser.add_argument('--num_workers', type=int, default=16,
                            help='num of workers to use')
        parser.add_argument('--batch_size', default='2', type=int,
                    help='batch size')
        parser.add_argument('--temprature', default='0.0001', type=float,
                            help='temprature')
        parser.add_argument('--base_temperature', default='0.001', type=float,
                            help='base_temperature')
        #888888888888888888888
        parser.add_argument('--lr', default='0.0001', type=float,
                    help='learning rate')
        parser.add_argument('--lr_method', default='Adam', type=str,
                    help='optimization method')
        parser.add_argument('--lr_decay', default='0.000005', type=float,
                    help='learning rate decay rate')
        parser.add_argument('--momentum', default='0.99', type=float,
                            help='momentum')
        parser.add_argument('--gamma', default='0.99', type=float,
                           help='gamma')
        parser.add_argument('--step_size', default='1256', type=int,
                            help='step_size')
        parser.add_argument('--encoder', default='bilstm', type=str,
                            help='编码器')

        #********************
        parser.add_argument('--clip', default='10', type=float,
                    help='gradient clipping')
        parser.add_argument('--l2_reg_lambda', default='0.01', type=float,
                    help='l2 regularization coefficient')
        parser.add_argument('--classifier_type', default='linear', type=str,
                            help='the type of classifier')
        parser.add_argument('--nepoch_no_imprv', default='5', type=int,
                            help='number of epoch patience')
        parser.add_argument('--num_heads', default='6', type=int,
                            help='number of attention head')
        parser.add_argument('--ffn_dim', default='2048', type=int,
                            help='number of attention head')
        parser.add_argument('--num_layer', default='1', type=int,
                            help='number of layer')
        # data and results paths
        parser.add_argument('--dir_output_sup', default='output', type=str,
                    help='directory for output sup')
        parser.add_argument('--dir_output', default='output', type=str,
                            help='directory for output')
        parser.add_argument('--dir_out_class', default='output/', type=str,
                            help='directory for output')
        parser.add_argument('--data_root', default='data_fold/', type=str,
                            help='directory for dataset')

        parser.add_argument('--data_name', default='moxifloxacin_2', type=str,
                            help='name for dataset')
        parser.add_argument('--folds', default='split_1', type=str,
                            help='name for dataset')

        # character embedding
        parser.add_argument('--embedding_file', default='data/bert_word_embeddings.txt',
                           type=str, help='directory for trimmed char embeddings file')

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
        # misc  没有default的情况下 如果是store_false,则默认值是True，如果是store_true,则默认值是False
        parser.add_argument('--restore', action='store_true',
                    help='whether restore from previous trained model')
        parser.add_argument('--random_embeddings', action='store_false',
                    help='whether use random embedding for characters')
        parser.add_argument('--train_accuracy', action='store_false',
                            help='whether report accuracy while training')

        # transformer
        parser.add_argument('--use_transformer', action='store_false',
                            help='whether use transformer for sentence representation')

        # att or not
        parser.add_argument('--use_attention', action='store_true',
                            help='whether use attention based pooling')
        # att or not
        parser.add_argument('--use_doc_attention', action='store_true',
                            help='whether use doc attention based pooling')


        # transformer
        parser.add_argument('--use_doc_transformer', action='store_false',
                            help='whether use transformer for doc representation')

        self.parser.parse_args(namespace=self)

        self.dir_model_sup  = os.path.join(self.dir_output_sup + '/' + self.data_name + '/' + self.folds)
        self.dir_model = os.path.join(self.dir_output + '/' + self.data_name + '/' + self.folds)
        self.path_log   = os.path.join(self.dir_output + '/' + self.data_name + '/' + self.folds, "log_cc"
                                                                                                  ".txt")
        # self.path_log1 = os.path.join(self.dir_output + '/' + self.data_name + '/' + self.folds, "log_sup"
        #                                                                                         ".txt")

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
        # logger1 = get_logger(self.path_log1)


        # log the attributes
        msg = ', '.join(['{}: {}'.format(attr, getattr(self, attr)) for attr in dir(self) \
                        if not callable(getattr(self, attr)) and not attr.startswith("__")])
        logger.info(msg)


        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """

        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)

        # 3. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 4. get pre-trained embeddings
        # if self.random_embeddings:
        #     print('Randomly initializes the character vector....')
        #     dim_word = 128
        #     self.embeddings = get_random_wordvec_vectors(dim_word, self.vocab_words)
        #     self.word_position_embedding = positional_embedding(12450, position='word', dim=dim_word)
        #     if self.use_transformer:
        #         self.sentence_position_embedding = positional_embedding(4009, position='sentence',
        #                                                                 dim=self.num_units)
        #
        # else:
        #     print('Using pre-embedding to initialize the character vector....')
        #     self.embeddings = get_trimmed_wordvec_vectors(self.embedding_file, self.vocab_words)
        #     self.dim_word = self.embeddings.shape[1]

