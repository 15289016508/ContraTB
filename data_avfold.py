import os
import random

fold_num = 10

def resplit_corpus(name):

    datalist = []
    with open(f'data_moxifloxacin/{name}/data.txt'.format(name), 'r', encoding='utf-8') as file:
        lines = file.readlines()
        length = len(lines)
        for line in lines:
            datalist.append(line.replace('\n',''))
    print(length) # 11942 12165 11075 12253

    wordlist = []
    with open(f'data_moxifloxacin/{name}/sentence_position.txt'.format(name), 'r', encoding='utf-8') as wordfile:
        wlines = wordfile.readlines()
        for wline in wlines:
            wordlist.append(wline.replace('\n',''))

    sentencelist = []
    with open(f'data_moxifloxacin/{name}/sentence_position.txt'.format(name), 'r', encoding='utf-8') as sentencefile:
        slines = sentencefile.readlines()
        for sline in slines:
            sentencelist.append(sline.replace('\n',''))

    x = [i for i in range(length)]
    # datad = dict(zip(x,datalist))
    # wordd = dict(zip(x,wordlist))
    # sentenced = dict(zip(x,sentencelist))
    # #
    # dict_key_ls = list(datad.keys())
    #

    positive_indices = [i for i, line in enumerate(datalist) if line.split(',')[1] == 'Y']
    negative_indices = [i for i, line in enumerate(datalist) if line.split(',')[1] == 'N']


    num_positive = len(positive_indices)
    num_negative = len(negative_indices)
    print("num_positive",num_positive)
    print("num_negative", num_negative)

    # 计算验证集和测试集的正负例数量
    num_pos_dev = int(num_positive * 0.1)
    num_neg_dev = int(num_negative * 0.1)

    num_pos_test = int(num_positive * 0.1)
    num_neg_test = int(num_negative * 0.1)

    # 计算训练集的正负例数量
    num_pos_train = num_positive - num_pos_dev - num_pos_test
    num_neg_train = num_negative - num_neg_dev - num_neg_test
    print()

    # 从正负例索引中随机选择数据


    # 创建文件夹并保存数据集
    for split_index in range(fold_num):
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        # 划分训练集、验证集和测试集的索引
        train_indices = positive_indices[:num_pos_train] + negative_indices[:num_neg_train]
        dev_indices = positive_indices[num_pos_train:num_pos_train + num_pos_dev] + negative_indices[
                                                                                    num_neg_train:num_neg_train + num_neg_dev]
        test_indices = positive_indices[
                       num_pos_train + num_pos_dev:num_pos_train + num_pos_dev + num_pos_test] + negative_indices[
                                                                                                 num_neg_train + num_neg_dev:num_neg_train + num_neg_dev + num_neg_test]

        # 重新排序索引
        train_indices.sort()
        dev_indices.sort()
        test_indices.sort()

        # 根据索引划分数据集
        data_train_list = [datalist[i] for i in train_indices]
        data_dev_list = [datalist[i] for i in dev_indices]
        data_test_list = [datalist[i] for i in test_indices]

        word_train_list = [wordlist[i] for i in train_indices]
        word_dev_list = [wordlist[i] for i in dev_indices]
        word_test_list = [wordlist[i] for i in test_indices]

        sentence_train_list = [sentencelist[i] for i in train_indices]
        sentence_dev_list = [sentencelist[i] for i in dev_indices]
        sentence_test_list = [sentencelist[i] for i in test_indices]
        corpus_dir = corpus_dir = f"data_fold/{name}/split_{split_index}"

        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)
        data_train_corpus_path = corpus_dir + '/' + 'train.txt'
        data_dev_corpus_path = corpus_dir + '/' + 'dev.txt'
        data_test_corpus_path = corpus_dir + '/' + 'test.txt'

        word_train_corpus_path = corpus_dir + '/' + 'train_wordposition.txt'
        word_dev_corpus_path = corpus_dir + '/' + 'dev_wordposition.txt'
        word_test_corpus_path = corpus_dir + '/' + 'test_wordposition.txt'

        sentence_train_corpus_path = corpus_dir + '/' + 'train_sentenceposition.txt'
        sentence_dev_corpus_path = corpus_dir + '/' + 'dev_sentenceposition.txt'
        sentence_test_corpus_path = corpus_dir + '/' + 'test_sentenceposition.txt'

        # 写入数据
        with open(data_train_corpus_path, 'w', encoding='utf-8') as train_file:
            for line in data_train_list:
                train_file.write(line + '\n')

        with open(data_dev_corpus_path,'w',encoding='utf-8') as dev_corpus_path:
            for line in data_dev_list:
                dev_corpus_path.write(line + '\n')

        with open(data_test_corpus_path, 'w', encoding='utf-8') as test_corpus_path:
            for line in data_test_list:
                test_corpus_path.write(line + '\n')

        with open(word_train_corpus_path, 'w', encoding='utf-8') as train_file:
            for line in word_train_list:
                train_file.write(line + '\n')

        with open(word_dev_corpus_path,'w',encoding='utf-8') as dev_corpus_path:
           for line in word_dev_list:
               dev_corpus_path.write(line + '\n')

        with open(word_test_corpus_path, 'w', encoding='utf-8') as test_corpus_path:
            for line in word_test_list:
                test_corpus_path.write(line + '\n')

        with open(sentence_train_corpus_path, 'w', encoding='utf-8') as train_file:
            for line in sentence_train_list:
                train_file.write(line + '\n')

        with open(sentence_dev_corpus_path,'w',encoding='utf-8') as dev_corpus_path:
           for line in sentence_dev_list:
               dev_corpus_path.write(line + '\n')

        with open(sentence_test_corpus_path, 'w', encoding='utf-8') as test_corpus_path:
            for line in sentence_test_list:
                test_corpus_path.write(line + '\n')

if __name__ == "__main__":
    for root, dirs, files in os.walk('data_moxifloxacin/'):
        for dir in dirs:
            print(dir)
            resplit_corpus(dir)
