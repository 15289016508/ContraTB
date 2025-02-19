import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_attention_zhao_compute import DocumentModel
# from  model_gradient import DocumentModel
import utils
import logging
from dataset import CustomDataset
from Optim_Choose import CustomOptimizer
from sklearn.metrics import roc_curve, auc
import os
from config import Config
import torch.nn.functional as F
import pandas as pd
# 配置文件和全局变量
config_list = Config()
gpu_id = config_list.gpu
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()

class ModelTrainer(nn.Module):
    def __init__(self, model, t_data, t_word_p, t_sentence_p, d_data, d_word_p, d_sentence_p, criterion, optimizer,
                 num_epochs, patience=3):
        super(ModelTrainer, self).__init__()
        self.model = model
        self.t_data = t_data
        self.t_word_position = t_word_p
        self.t_sentence_position = t_sentence_p
        self.d_data = d_data
        self.d_word_position = d_word_p
        self.d_sentence_position = d_sentence_p
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience
        self.num_epochs = num_epochs
        self.model.to(device)

    def tensor_collate(self, batch):
        x_batch = [item[0] for item in batch]
        y_batch = [item[1] for item in batch]
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.long).squeeze()
        word_position_batch = [item[2] for item in batch]
        sentence_batch = [item[3] for item in batch]
        document_lengths = [len(doc) for doc in x_batch]
        word_ids, sentence_lengths = utils.pad_sequences(x_batch, pad_tok=0, nlevels=2)
        truncated_sentence_lengths = [[length for length in lengths if length > 0] for lengths in sentence_lengths]
        flattened_sentence_lengths = [length for lengths in truncated_sentence_lengths for length in lengths]
        word_position_ids, _ = utils.pad_sequences(word_position_batch, pad_tok=0, nlevels=2)
        sentence_position_ids, _ = utils.pad_sequences(sentence_batch, pad_tok=0, nlevels=1)
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
        position_idf_tensor = torch.tensor(word_position_ids, dtype=torch.long)
        sentence_idf_tensor = torch.tensor(sentence_position_ids, dtype=torch.long)
        word_ids_tensor = word_ids_tensor.to(device)
        position_idf_tensor = position_idf_tensor.to(device)
        sentence_idf_tensor = sentence_idf_tensor.to(device)
        y_batch_tensor = y_batch_tensor.to(device)

        return word_ids_tensor, y_batch_tensor, position_idf_tensor, sentence_idf_tensor, document_lengths, flattened_sentence_lengths

    def train(self):
        writer = SummaryWriter(log_dir='logs')
        best_f1 = 0
        num_bad_epochs = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            custom_dataset = CustomDataset(self.t_data, self.t_word_position, self.t_sentence_position)
            data_loader = tqdm(DataLoader(dataset=custom_dataset, batch_size=config_list.batch_size,
                                          collate_fn=self.tensor_collate, shuffle=True))
            for x_batch, y_batch, position_batch, sentence_batch, document_lengths, max_sentence_length in data_loader:
                self.optimizer.zero_grad()
                x_batch = x_batch.to(device)
                position_batch = position_batch.to(device)
                sentence_batch = sentence_batch.to(device)
                y_batch = y_batch.to(device)
                if y_batch.numel() > 1:
                    y_pred = self.model(x_batch, position_batch, sentence_batch, document_lengths, max_sentence_length)
                    y_pred = y_pred.to(device)
                    y_batch = y_batch.type(torch.long)
                    loss = self.criterion(y_pred, y_batch)
                    writer.add_scalar('Train/Loss', loss, epoch)
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f'Parameter: {name}, Gradient: {param.grad.sum()}')
                    #     else:
                    #         print(f'Parameter: {name}, Gradient: None')
                    self.optimizer.step(epoch)
                    total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Training Loss: {avg_loss}")
            logger.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Average Epoch Loss: {avg_loss:.4f}')
            print("开始验证模型：")
            val_loss, val_accuracy, val_precision, val_recall, val_f1, AUC = self.test(self.d_data,
                                                                                       self.d_word_position,
                                                                                       self.d_sentence_position)
            print(
                f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                num_bad_epochs = 0
                logger.info("- new best score!")
                model_save_path = os.path.join(config_list.dir_model, 'last_attention.pth')
                torch.save(self.model.state_dict(), model_save_path)
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= config_list.nepoch_no_imprv:
                    print(f"Early stopping after {num_bad_epochs + 1} epochs without improvement in F1.")
                    break
        writer.close()


    def test(self, data, word_position, sentence_position,top_n_sentences=2, top_n_words=2):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_probs = []
        all_labels = []
        custom_dataset = CustomDataset(data, word_position, sentence_position)
        data_loader = tqdm(DataLoader(dataset=custom_dataset, batch_size=config_list.batch_size,
                                     collate_fn=self.tensor_collate, shuffle=False))
        with open("attention_m_2", 'w') as f:
            with torch.no_grad():
                num_classes = 2  # 根据你的实际分类类别数目设置
                confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
                for x_batch, y_batch, position_batch, sentence_batch, document_lengths, max_sentence_length in data_loader:
                    x_batch = x_batch.to(device)
                    position_batch = position_batch.to(device)
                    sentence_batch = sentence_batch.to(device)
                    print(sentence_batch)
                    y_batch = y_batch.to(device)
                    y_pred ,attention_w,attention_d= self.model(x_batch, position_batch, sentence_batch, document_lengths, max_sentence_length)
                    y_pred = y_pred.to(device)
                    print(y_pred)
                    print(y_batch)
                    attention_w = attention_w.unsqueeze(0)
                    y_pred_cl = y_pred.argmax(dim=1)
                    print(y_pred)
                    is_positive = (y_pred_cl == 1) & (y_batch == 1)
                    print(is_positive)
                    if is_positive.sum().item() == 0:
                        continue
                    result = ''
                    print(attention_d.size(0))
                    for i in range(attention_d.size(0)):
                        current_attention_d = attention_d[i]
                        current_attention_w = attention_w[i]
                        print("current_attention_d ", current_attention_d)
                        print("current_attention_w ", current_attention_w)
                        # 找到注意力值最高的前 top_n_sentences 个句子的索引
                        valid_sentence_indices = (current_attention_d > 0.01).nonzero(as_tuple=False).view(-1)
                        k_sentences = min(top_n_sentences, valid_sentence_indices.size(0))
                        topk_indices = valid_sentence_indices[
                            current_attention_d[valid_sentence_indices].topk(k_sentences).indices]
                        print(topk_indices)
                        print(f"Sample {i} - top {k_sentences} sentence indices: {topk_indices.tolist()}")
                        for idx in topk_indices:
                            # 获取对应句子的注意力权重和真实索引
                            sentence_attention_w = current_attention_w[idx]
                            sentence_real_index = sentence_batch[i, idx].item()
                            sentence_attention_score = current_attention_d[idx].item()
                            print(
                                f"  Sentence index {idx} (real index {sentence_real_index}) - Attention score: {sentence_attention_score}")

                            # 找到该句子中注意力值最高的前 top_n_words 个词的索引
                            non_zero_count = (sentence_attention_w != 0).sum().item()
                            k = min(top_n_words, non_zero_count)
                            if k > 0:
                                topm_word_indices = sentence_attention_w.topk(k).indices
                                print("topm_word_indices", topm_word_indices)

                                for word_idx in topm_word_indices:
                                    # 获取词的真实索引和注意力分数
                                    word_real_index = position_batch[i, idx, word_idx].item()
                                    print("sentence_attention_w", sentence_attention_w)
                                    word_attention_score = sentence_attention_w[word_idx].item()
                                    if word_attention_score == 0.0:
                                        word_real_index = -1
                                    # 格式化输出
                                    # result_string = f"{sentence_real_index}-{sentence_attention_score}-{word_real_index}-{word_attention_score}"
                                    result_string = f"{sentence_real_index}-{word_real_index}"
                                    # result_string = f"{sentence_real_index}"
                                    # print(
                                    #     f"    Word index {word_idx} (real index {word_real_index}) - Attention score: {word_attention_score}")
                                    result = result + result_string + ","

                                    print(f"    Result: {result_string}")
                    f.write(result + "\n")

        return 0


    def find_largest_frequent_itemsets(self, data, min_support=0.01):
        """
        Finds the largest frequent itemsets and calculates confidence, support, and lift.

        Parameters:
        - data: List of transactions (list of lists of items)
        - min_support: Minimum support threshold for frequent itemsets

        Returns:
        - List of largest frequent itemsets
        """
        # Initialize TransactionEncoder
        data = [line.strip().split(',') for line in data]
        # Remove empty strings from transactions
        cleaned_data = [[item for item in transaction if item] for transaction in data]
        te = TransactionEncoder()
        # Transform your data
        te_ary = te.fit(cleaned_data).transform(cleaned_data)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Find frequent itemsets with minimum support
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

        # Calculate the length of each itemset
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        print("frequent_itemsets", frequent_itemsets)
        # Find the maximum length of itemsets
        max_length = frequent_itemsets['length'].max()

        # Filter itemsets based on the maximum length
        largest_frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] == max_length]
        print("largest_frequent_itemsets", largest_frequent_itemsets)

        # Generate association rules from frequent itemsets
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0)

        # Filter rules to only include those with the largest frequent itemsets as consequents
        largest_itemsets_set = set(largest_frequent_itemsets['itemsets'])
        filtered_rules = rules[rules['consequents'].apply(lambda x: x in largest_itemsets_set)]
        filtered_rules = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

        # Filter rules to include only those with lift > 1
        filtered_rules_lift_gt_1 = filtered_rules[filtered_rules['lift'] > 1]

        print("filtered_rules with lift > 1", filtered_rules_lift_gt_1)

        # Write the results to a file
        with open("frequent_itemsets_m_2", "w") as f:
            f.write("Frequent itemsets\n")
            f.write(frequent_itemsets.to_string(index=False))
            f.write("\n\n")
            f.write("Largest frequent itemsets\n")
            f.write(largest_frequent_itemsets.to_string(index=False))
            f.write("\n\n")
            f.write("Association rules\n")
            f.write(filtered_rules.to_string(index=False))
            f.write("\n\n")
            f.write("Association rules with lift > 1\n")
            f.write(filtered_rules_lift_gt_1.to_string(index=False))

        return largest_frequent_itemsets['itemsets'].tolist(), filtered_rules_lift_gt_1

model = DocumentModel(config_list).to(device)
optimizer = CustomOptimizer(model.parameters(),config_list)
criterion = nn.CrossEntropyLoss()

print("加载验证集：")
dev = utils.Dataset(config_list.filename_dev,
                  config_list.processing_word,
                  config_list.processing_tag)
# print("加载训练集：")
train = utils.Dataset(config_list.filename_train,
                    config_list.processing_word,
                    config_list.processing_tag)
# for sentences, tags in train:
#     print(tags,"******",sentences)
print("加载测试集：")
test = utils.Dataset(config_list.filename_test,
                   config_list.processing_word,
                   config_list.processing_tag)

train_word_position = utils.positional_encoding(config_list.word_position_train, position='word')
train_sentence_position = utils.positional_encoding(config_list.sentence_position_train, position='sentence')
dev_word_position = utils.positional_encoding(config_list.word_position_dev, position='word')
dev_sentence_position = utils.positional_encoding(config_list.sentence_position_dev, position='sentence')
test_word_position = utils.positional_encoding(config_list.word_position_test, position='word')
test_sentence_position = utils.positional_encoding(config_list.sentence_position_test, position='sentence')

trainer = ModelTrainer(model, train, train_word_position, train_sentence_position,dev, dev_word_position, dev_sentence_position, criterion, optimizer , num_epochs=config_list.epochs).to(device)
print("begin to call train()")
# trainer.train()

logger.info("begin to test")
model.load_state_dict(torch.load(os.path.join(config_list.dir_model,'last_attention.pth')),strict=False)
trainer.test(test,test_word_position,test_sentence_position)
# trainer.test_gradient(test,test_word_position,test_sentence_position)
with open("attention_m_2", 'r') as f:
    data = f.readlines()
a = trainer.find_largest_frequent_itemsets(data)