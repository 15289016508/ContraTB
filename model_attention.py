import torch.nn as nn
import torch
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class DocumentModel(nn.Module):
    def __init__(self, config):
        super(DocumentModel, self).__init__()
        gpu_id = config.gpu
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm_lstm_w = nn.LayerNorm(2 * config.hidden_size_lstm_sentence)  # Add LayerNorm after word-level LSTM
        self.layer_norm_lstm_d = nn.LayerNorm(2 * config.hidden_size_lstm_sentence)  # Add LayerNorm after document-level LSTM
        self.bilstm_w = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_size_lstm_sentence,
            num_layers=config.num_layer,
            dropout=config.dropout,
            bidirectional=True,
            batch_first=True
        )

        self.bilstm_d = nn.LSTM(
            input_size=300,
            hidden_size=config.hidden_size_lstm_sentence,
            num_layers=config.num_layer,
            dropout=config.dropout,
            bidirectional=True,
            batch_first=True
        )

        dim_word = 128
        self.character_embedding = nn.Embedding(len(config.vocab_words), dim_word, padding_idx=0)
        self.positional_embedding = nn.Embedding(12450, dim_word, padding_idx=0)
        self.sentence_position_embedding = nn.Embedding(4009, 300, padding_idx=0)
        self.init_embedding(self.positional_embedding, dim_word, 12450)
        self.init_embedding(self.sentence_position_embedding, 300, 4009)
        self.w = nn.Parameter(torch.Tensor(2 * config.hidden_size_lstm_sentence, config.ntags))
        self.b = nn.Parameter(torch.Tensor(config.ntags))
        self.w_word = nn.Parameter(torch.Tensor(2 * config.hidden_size_lstm_sentence, config.attention_size))
        self.b_word = nn.Parameter(torch.Tensor(config.attention_size))
        self.u_word = nn.Parameter(torch.Tensor(config.attention_size, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * config.hidden_size_lstm_sentence, config.attention_size))
        self.b_s = nn.Parameter(torch.Tensor(config.attention_size))
        self.u_s = nn.Parameter(torch.Tensor(config.attention_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.w_word)
        nn.init.xavier_uniform_(self.u_word)
        nn.init.zeros_(self.b_word)
        nn.init.xavier_uniform_(self.w_s)
        nn.init.xavier_uniform_(self.u_s)
        nn.init.zeros_(self.b_s)
        nn.init.xavier_uniform_(self.character_embedding.weight)

    def init_embedding(self, embedding_layer, dim_word, max_len):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_word, 2).float() * -(math.log(10000.0) / dim_word))
        sin_values = torch.sin(position * div_term)
        cos_values = torch.cos(position * div_term)
        embedding_layer.weight.data[:, 0::2] = sin_values
        embedding_layer.weight.data[:, 1::2] = cos_values

    def forward(self, word_ids_tensor, position_idf_tensor, sentence_idf_tensor, document_lengths, sentence_length):
        word_ids_tensor = word_ids_tensor.to(self.device)
        position_idf_tensor = position_idf_tensor.to(self.device)
        sentence_idf_tensor = sentence_idf_tensor.to(self.device)
        word_embedding = self.character_embedding(word_ids_tensor)
        position_embedding = self.positional_embedding(position_idf_tensor)
        sentence_embedding = self.dropout(self.sentence_position_embedding(sentence_idf_tensor))
        combined_data = word_embedding + position_embedding
        batch_size, max_sentences, word_num, hidden_size = combined_data.size()
        character_embeddings = self.dropout(combined_data.float())
        s = character_embeddings.shape
        # print("character_embeddings", character_embeddings.shape)
        new_char = torch.cat([character_embeddings[i][:document_lengths[i]] for i in range(batch_size)], dim=0)
        # print("new_char", new_char.shape)
        packed_input = pack_padded_sequence(new_char, sentence_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm_w(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        U_sent =  torch.tanh(torch.matmul(output, self.w_word) + self.b_word)
        # 构造掩码张量
        max_length = max(sentence_length)
        mask_tensor = torch.zeros(len(sentence_length), max_length, dtype=torch.bool).to(self.device)
        for i, length in enumerate(sentence_length):
            mask_tensor[i, length:] = True
        # 将掩码应用于注意力分数
        masked_attention_scores = torch.matmul(U_sent, self.u_word).view(-1, s[2]).masked_fill(mask_tensor,
                                                                                    float("-inf"))
        # print("masked_attention_scores_word", masked_attention_scores)
        # 计算注意力权重
        A = F.softmax(masked_attention_scores, dim=-1)
        attention_w = A
        # print(A.shape)
        # print("A", A)
        output = output.view(-1, s[2], 300)
        # print(output.shape)
        output = torch.sum(output * A.unsqueeze(-1).expand_as(output), dim=1)
        output= self.layer_norm_lstm_w(output)
        # print("lstm_output_word_attention", output.shape)
        start_idx = 0
        sentence_outputs = []
        for i, length in enumerate(document_lengths):
            end_idx = start_idx + length
            sentence_outputs.append(output[start_idx:end_idx, :])
            start_idx = end_idx
        max_sentences = max(document_lengths)
        padded_sentence_outputs = [
            torch.cat([s, torch.zeros(max_sentences - s.size(0), s.size(1)).to(self.device)], dim=0) if s.size(
                0) < max_sentences else s for s in sentence_outputs]
        sliced_outputs = torch.stack(padded_sentence_outputs).to(self.device)

        reshaped_outputs = sliced_outputs.view(batch_size, max_sentences, 300)

        sentence_representation = sentence_embedding + reshaped_outputs
        # print("sentence_representation", sentence_representation.shape)

        sentence_representation = pack_padded_sequence(sentence_representation, document_lengths, batch_first=True,
                                                       enforce_sorted=False)

        packed_document_representation, _ = self.bilstm_d(sentence_representation)

        output, _ = pad_packed_sequence(packed_document_representation, batch_first=True)
        U_sent =  torch.tanh(torch.matmul(output, self.w_s) + self.b_s)
        # print("output", output.shape)
        # print("U_sent", U_sent.shape)
        # 构造掩码张量
        max_length = max(document_lengths)
        mask_tensor = torch.zeros(len(document_lengths), max_length, dtype=torch.bool).to(self.device)
        for i, length in enumerate(document_lengths):
            mask_tensor[i, length:] = True
        # 将掩码应用于注意力分数
        masked_attention_scores = torch.matmul(U_sent, self.u_s).view(-1, s[1]).masked_fill(mask_tensor, float("-inf"))
        # print("masked_attention_scores_word", masked_attention_scores)
        # 计算注意力权重
        A = F.softmax(masked_attention_scores, dim=-1)
        attention_d = A
        # print(A.shape)
        # print("A", A)
        output = output.view(-1, s[1], 300)
        # print(output.shape)
        output = torch.sum(output * A.unsqueeze(-1).expand_as(output), dim=1)
        # print("lstm_output_sentence_attention", output.shape)
        output = self.layer_norm_lstm_d(output)
        document_representation = self.dropout(output)
        pred = torch.matmul(document_representation, self.w) + self.b
        output = pred.view(-1, self.w.size(1))
        # output = self.Classifier(document_representation)

        return output
