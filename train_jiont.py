import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_attention import DocumentModel
import utils
import logging
from dataset import CustomDataset
from Optim_Choose import CustomOptimizer
from sklearn.metrics import roc_curve, auc
import os
from configcl import Config
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
                    self.optimizer.step( )
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

            # 打印当前学习率
            logger.info(f"Current learning rate: {self.optimizer.scheduler.get_last_lr()}")

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

    def test(self, data, word_position, sentence_position):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_probs = []
        all_labels = []
        custom_dataset = CustomDataset(data, word_position, sentence_position)
        data_loader = tqdm(DataLoader(dataset=custom_dataset, batch_size=config_list.batch_size,
                                      collate_fn=self.tensor_collate, shuffle=False))
        with torch.no_grad():
            num_classes = 2  # 根据你的实际分类类别数目设置
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
            for x_batch, y_batch, position_batch, sentence_batch, document_lengths, max_sentence_length in data_loader:
                x_batch = x_batch.to(device)
                position_batch = position_batch.to(device)
                sentence_batch = sentence_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = self.model(x_batch, position_batch, sentence_batch, document_lengths, max_sentence_length)
                y_pred = y_pred.to(device)

                y_batch = y_batch.type(torch.long)
                if y_batch.numel() == 1:
                    y_batch = y_batch.unsqueeze(0)

                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)

                for pred, act in zip(predicted.cpu().numpy(), y_batch.cpu().numpy()):
                    confusion_matrix[act, pred] += 1

                correct_predictions += (predicted == y_batch).sum().item()
                total_predictions += len(y_batch)

                y_probs = F.softmax(y_pred, dim=1)
                all_probs.extend(y_probs[:, 1].cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix)

        avg_loss = total_loss / len(data_loader)
        tn, fp, fn, tp = confusion_matrix.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        sensitivity_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision_pos * sensitivity_pos) / (precision_pos + sensitivity_pos) if (
                                                                                                            precision_pos + sensitivity_pos) > 0 else 0

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        # print("all_probs",all_probs)
        # print("all_labels", all_labels)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        auc_score = auc(fpr, tpr)

        logger.info(f"Acc : {accuracy}")
        logger.info(f"敏感性_sensitivity : {sensitivity_pos}")
        logger.info(f"特异性_specificity: {specificity_neg}")
        logger.info(f"阳性预测值_PPV: {precision_pos}")
        logger.info(f"阴性预测值_NPV: {precision_neg}")
        logger.info(f"Recall all: {sensitivity_pos}")
        logger.info(f"F1 all : {f1_score}")
        logger.info(f"auc_score: {auc_score}")
        # 保存结果到指定目录中的Excel文件
        results = {
            "Dataset": [f"{config_list.data_name}"],
            "lr": [f"{config_list.lr}"],
            "lr_decay": [f"{config_list.lr_decay}"],
            "Split": [f"{config_list.folds}"],
            "Accuracy": [accuracy * 100],
            "Sensitivity": [sensitivity_pos * 100],
            "Specificity": [specificity_neg * 100],
            "PPV": [precision_pos * 100],
            "NPV": [precision_neg * 100],
            "F1 Score": [f1_score * 100],
            "AUC Score": [auc_score * 100]
        }

        df = pd.DataFrame(results)
        df = df.round(2)
        output_path = os.path.join(config_list.dir_output, f"{config_list.data_name}_{config_list.lr_decay}_{config_list.lr}.xlsx")
        # df.to_excel(output_path, index=False, header=True)
        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # 保留每个 Split 下的最后一次结果
            combined_df = combined_df.drop_duplicates(subset=["Split"], keep='last')
        else:
            combined_df = df

        combined_df.to_excel(output_path, index=False, header=True)
        return avg_loss, accuracy, precision_pos, sensitivity_pos, f1_score, auc_score

    def load_pretrained_bilstm_attention(self, path):
        pre_trained_path = path  # 替换为你的预训练模型路径
        # 加载预训练模型，并将其映射到指定的 GPU
        pretrained_dict = torch.load(pre_trained_path, map_location=f"cuda:{gpu_id}")
        model_dict = self.model.state_dict()

        # 过滤掉分类层的参数，只保留BiLSTM+注意力层的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新现有的 model_dict
        model_dict.update(pretrained_dict)

        # 加载我们真正需要的参数
        self.model.load_state_dict(model_dict)

        # 随机初始化分类器参数
        nn.init.xavier_uniform_(self.model.w)
        nn.init.normal_(self.model.b, mean=0, std=0.1)


model = DocumentModel(config_list)
optimizer = CustomOptimizer(model.parameters(),config_list)
criterion = nn.CrossEntropyLoss()
# class_weights = torch.tensor([2, 10], dtype=torch.float32)
# # 将 class_weights 移动到 'cuda' 上
# class_weights = class_weights.to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights.clone().detach())
print("加载验证集：")
dev = utils.Dataset(config_list.filename_dev,
                  config_list.processing_word,
                  config_list.processing_tag)
print("加载训练集：")
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

trainer = ModelTrainer(model, train, train_word_position, train_sentence_position,dev, dev_word_position, dev_sentence_position, criterion, optimizer , num_epochs=config_list.epochs)
print("begin to call train()")
pretrained_path = f"output_sup/{config_list.data_name}/{config_list.folds}/lastmodel_{config_list.folds}.pt"
# pretrained_state_dict = torch.load(pretrained_path)
trainer.load_pretrained_bilstm_attention(pretrained_path)  # 加载预训练权重，但分类器层参数保持随机初始化
trainer.train()

logger.info("begin to test")
model.load_state_dict(torch.load(os.path.join(config_list.dir_model,'last_attention.pth')))
trainer.test(test,test_word_position,test_sentence_position)
