import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse, utils
from model_sup import DocumentModel
from sup_losses import SupConLoss
from config_sup_simple import Config
from dataset import CustomDataset
from Optim_Choose import CustomOptimizer
from torch.utils.data import DataLoader
import os, logging
from Sampler_seq import CustomBatchSampler

parser = argparse.ArgumentParser(description='Argument for training')


# multiprocessing.set_start_method('spawn')
# Define your SentenceTransformer, DocumentTransformer, LinearClassifier, and DocumentModel classes here...

class ModelTrainer:
    # selected_gpu = 1
    def __init__(self, model, data, word_p, sentence_p, num_epochs, patience=1000):
        self.model = model
        self.data = data
        self.word_position = word_p
        self.sentence_position = sentence_p
        # self.criterion = criterion
        # self.optimizer = optimizer
        self.patience = patience
        self.num_epochs = num_epochs
        # self.batchsize=28
        # self.embedding_model = EmbeddingModel(config_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.patience = patience
        self.num_epochs = num_epochs

    def tensor_collate(self, batch):
        x_batch = [item[0] for item in batch]
        y_batch = [item[1] for item in batch]
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.long).squeeze()  # 调整为一维张量
        word_positon_batch = [item[2] for item in batch]
        sentence_batch = [item[3] for item in batch]
        # print("x_batch.shape",x_batch)
        document_lengths = [len(doc) for doc in x_batch]
        # print("document_lengths",document_lengths)
        max_document_length = max(document_lengths)
        # print("max_document_length",max_document_length)
        word_ids, sentence_lengths = utils.pad_sequences(x_batch, pad_tok=0, nlevels=2)
        # print("sentence_lengths",sentence_lengths)

        # 截断列表中0开始的部分
        truncated_sentence_lengths = [[length for length in lengths if length > 0] for lengths in sentence_lengths]
        flattened_sentence_lengths = [length for lengths in truncated_sentence_lengths for length in lengths]
        word_position_ids, _ = utils.pad_sequences(word_positon_batch, pad_tok=0, nlevels=2)
        # print("word_position_ids",word_position_ids)
        sentence_position_ids, _ = utils.pad_sequences(sentence_batch, pad_tok=0, nlevels=1)
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)  # 转换为float32
        # print("word_ids_tensor:", word_ids_tensor.shape)
        position_idf_tensor = torch.tensor(word_position_ids, dtype=torch.long)  # 转换为float32
        # print("position_idf_tensor :", position_idf_tensor.shape)
        sentence_idf_tensor = torch.tensor(sentence_position_ids, dtype=torch.long)  # 转换为float32
        # print("sentence_idf_tensor",sentence_idf_tensor.shape)
        word_ids_tensor = word_ids_tensor.to(self.device)
        position_idf_tensor = position_idf_tensor.to(self.device)
        sentence_idf_tensor = sentence_idf_tensor.to(self.device)
        y_batch_tensor = y_batch_tensor.to(self.device)
        # del word_ids_tensor, position_idf_tensor, sentence_idf_tensor
        return word_ids_tensor, y_batch_tensor, position_idf_tensor, sentence_idf_tensor, document_lengths, flattened_sentence_lengths

    def train(self):
        output_dir = config_list.dir_model
        # os.makedirs(output_dir, exist_ok=True)
        # Set up logging to a file in the output directory
        # log_filename = os.path.join(output_dir, 'training_log.txt')
        logging.basicConfig(filename=config_list.path_log, level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
        best_loss = 1000
        num_bad_epochs = 0

        for epoch in range(self.num_epochs):

            self.model.train()
            writer = SummaryWriter(log_dir='logs')
            total_loss = 0.0
            custom_dataset = CustomDataset(self.data, self.word_position, self.sentence_position)

            # 随机过采样得到不同的负例和正例，正例远远大于负例
            custom_batch_sampler = CustomBatchSampler(custom_dataset, batch_size=config_list.batch_size)

            data_loader = tqdm(DataLoader(dataset=custom_dataset,
                                          collate_fn=self.tensor_collate, batch_sampler=custom_batch_sampler))

            iter_num = 0

            for x_batch, y_batch, position_batch, sentence_batch, document_lengths, max_sentence_length in data_loader:
                # print(y_batch)
                # print("each batch")
                optimizer.zero_grad()
                doc_embedding = self.model(x_batch, position_batch, sentence_batch, document_lengths,
                                           max_sentence_length)

                labels = y_batch
                # print(doc_embedding.shape)

                # print("labels",labels)
                # Compute the loss
                contrastive_loss = criterion(doc_embedding, labels)
                # l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
                # for param in self.model.parameters():
                #     l2_reg += torch.norm(param, p=2)
                #
                # loss = contrastive_loss + config_list.l2_lambda * l2_reg
                # print("contrastive_loss",contrastive_loss)
                # 执行反向传播

                contrastive_loss.backward()
                # parameters_to_clip = [param for name, param in model.named_parameters() if
                #                     name.startswith('sentence_transformer')]

                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=4.0)

                # 在backward之后打印梯度
                # print('After backward:')
                # for name, param in model.named_parameters():
                #    if param.grad is not None:
                #        print(f'{name}: {param.grad.norm().item()}')
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f'Parameter name: {name}, Shape: {param.shape}, Grad norm: {param.grad.norm().item()}')
                #     else:
                #         print(f'Parameter name: {name}, Shape: {param.shape}, Grad: None')

                # print("contrastive_loss.item()",contrastive_loss.item(),"contrastive_loss",contrastive_loss)
                total_loss += contrastive_loss.item()

                current_learning_rate = optimizer.param_groups[0]['lr']
                # print("learning rate:", current_learning_rate)
                # print(f'Epoch [{epoch + 1}/{self.num_epochs}], Iteration [{iter_num}/{len(data_loader)}], '
                #       f'{iter_num}  Average Batch Loss: {contrastive_loss.item():.4f}')

                # Print loss for every accumulation step

                optimizer.step()
                # self.optimizer.zero_grad()  # Reset gradients after accumulation

                torch.cuda.empty_cache()

            average_epoch_loss = total_loss / len(data_loader)
            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                num_bad_epochs = 0
                model_save_path = os.path.join(output_dir,
                                               f'a-lr({epoch})t{config_list.temprature}-bs{config_list.batch_size}-{config_list.step_size}-{config_list.lr}-ga{config_list.gamma}-{config_list.data_name}_{config_list.folds}.pt')
                torch.save(self.model.state_dict(), model_save_path)
                print(f'Model saved at: {model_save_path}')
            else:
                num_bad_epochs += 1
                if num_bad_epochs >= 20:
                    print(f"Early stopping after {num_bad_epochs} epochs without improvement loss.")
                    break

            # print(f'Epoch [{epoch + 1}/{self.num_epochs}], Average Epoch Loss: {average_epoch_loss:.4f}')
            # lr_scheduler.step(average_epoch_loss)

            # Save the model every 2 epochs

            # output_dir=output_dir+'/t007'

            # logging.info('Final gradients after epoch:',epoch + 1)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         logging.info(f'{name}: {param.grad.norm().item()}')
            torch.cuda.empty_cache()
            # Log epoch information
            logging.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Average Epoch Loss: {average_epoch_loss:.4f},Learnin:{current_learning_rate}')



config_list = Config()
model = DocumentModel(config_list)

# optimizer = optim.Adam(model.parameters(), lr=config_list.lr)
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.8, verbose=True)


optimizer = CustomOptimizer(model.parameters(), config_list)
criterion = SupConLoss(config_list)

print("加载训练集：")
train = utils.Dataset(config_list.filename_train,
                      config_list.processing_word,
                      config_list.processing_tag)
train_word_position = utils.positional_encoding(config_list.word_position_train, position='word')
train_sentence_position = utils.positional_encoding(config_list.sentence_position_train, position='sentence')
# 创建并初始化训练器
trainer = ModelTrainer(model, train, train_word_position, train_sentence_position,
                       num_epochs=config_list.epochs)

# 调用训练函数
trainer.train()
