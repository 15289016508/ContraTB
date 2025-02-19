from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, word_position, sentence_position):
        self.data = data
        self.sentences = []  # 初始化 sentences 列表
        self.tags = []  # 初始化 tags 列表
        self.w_positions=[]
        self.s_positions=[]
        for sentences, tags in data:
            self.sentences.append(sentences)
            self.tags.append(tags)
        for word_positions in word_position:
            self.w_positions.append(word_positions)
        for sentence_positions in sentence_position:
            self.s_positions.append(sentence_positions)
        # # print("Number of samples:", len(self.data))
        # # print("Length of sentences list:", len(self.sentences))
        # # print("Length of tags list:", len(self.tags))
        # # print("Length of word_positions list:", len(self.w_positions))
        # # print("Length of sentence_positions list:", len(self.s_positions))
        # #
        # # # 打印示例
        # # print("Example sentence:", self.sentences[0])
        # # print("Example tag:", self.tags[0])
        # # print("Example word_position:", self.w_positions[0])
        # print("Example sentence_position:", self.s_positions[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print("Inside __getitem__")  # 添加调试输出
        # print(index)
        y = self.tags[index]
        # print(y)
        x= self.sentences[index]

        word_position = self.w_positions[index]
        sentence_position = self.s_positions[index]
        # print(f"Sample {index + 1} - Length of sentence: {x}")
        # print(f"Sample {index + 1} - Length of tag: {y}")
        # print(f"Sample {index + 1} - Length of word_position: {word_position}")
        # print(f"Sample {index + 1} - Length of sentence_position: {sentence_position}")
        return x, y, word_position, sentence_position
