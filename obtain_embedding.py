import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from dataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_collate(batch):
    x_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    y_batch_tensor = torch.tensor(y_batch, dtype=torch.long).squeeze()  # 调整为一维张量
    word_positon_batch = [item[2] for item in batch]
    sentence_batch = [item[3] for item in batch]

    document_lengths = [len(doc) for doc in x_batch]
    max_document_length = max(document_lengths)

    word_ids, sentence_lengths = utils.pad_sequences(x_batch, pad_tok=0, nlevels=2)

    truncated_sentence_lengths = [[length for length in lengths if length > 0] for lengths in sentence_lengths]
    flattened_sentence_lengths = [length for lengths in truncated_sentence_lengths for length in lengths]

    word_position_ids, _ = utils.pad_sequences(word_positon_batch, pad_tok=0, nlevels=2)
    sentence_position_ids, _ = utils.pad_sequences(sentence_batch, pad_tok=0, nlevels=1)

    word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
    position_idf_tensor = torch.tensor(word_position_ids, dtype=torch.long)
    sentence_idf_tensor = torch.tensor(sentence_position_ids, dtype=torch.long)

    word_ids_tensor = word_ids_tensor.to(device)
    position_idf_tensor = position_idf_tensor.to(device)
    sentence_idf_tensor = sentence_idf_tensor.to(device)
    y_batch_tensor = y_batch_tensor.to(device)

    return word_ids_tensor, y_batch_tensor, position_idf_tensor, sentence_idf_tensor, document_lengths, flattened_sentence_lengths


def obtain_embedding(model, model_path, data, word_position, sentence_position):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
        print("加载模型文件：", model_path)
    except FileNotFoundError:
        print(f"警告：未找到模型文件，将使用当前模型")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    embeddings = []
    labels = []
    custom_dataset = CustomDataset(data, word_position, sentence_position)
    data_loader = tqdm(DataLoader(dataset=custom_dataset, batch_size=20, collate_fn=tensor_collate))

    with torch.no_grad():
        for x_batch, y_batch, position_batch, sentence_batch, document_lengths, max_sentence_length in data_loader:
            doc_embedding = model(x_batch, position_batch, sentence_batch, document_lengths, max_sentence_length)
            embeddings.append(doc_embedding.cpu().numpy())  # Move to CPU before converting to NumPy
            labels.append(y_batch.cpu().numpy())  # Move to CPU before converting to NumPy

    embeddings = np.vstack(embeddings)

    # 将labels列表展平并转换为整数
    labels = np.concatenate(labels)
    labels_list = labels.tolist()

    return embeddings, labels_list
