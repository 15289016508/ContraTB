from config_mix import Config
from utils_mix import Dataset,get_processing_word,get_vocabs,get_all_name,\
    UNK, write_vocab,NUM
import argparse

parser = argparse.ArgumentParser()


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(parser, load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = Dataset(config.filename_dev, processing_word)
    test  = Dataset(config.filename_test, processing_word)
    train = Dataset(config.filename_train, processing_word)


    # # Build Word and Tag vocab
    vocab_words, vocab_tags, vocab_drugs,vocab_smiles= get_all_name([train, dev, test])
    # vocab_words, vocab_tags = get_vocabs([train])
    # vocab_glove = get_wordvec_vocab(config.filename_wordvec)

    #vocab = vocab_words & vocab_glove
    vocab = list(vocab_words)
    vocab.insert(0, UNK)
    vocab.append(NUM)

    #Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)
    write_vocab(vocab_drugs, config.filename_drugs)
    write_vocab(vocab_smiles, config.filename_smiles)

    # # Trim GloVe Vectors
    # vocab = load_vocab(config.filename_words)
    # print(vocab)
    #export_trimmed_wordvec_vectors(vocab, config.filename_wordvec,
     #                           config.filename_wordvec_trimmed)


if __name__ == "__main__":
    main()
