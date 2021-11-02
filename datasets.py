import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    创建dataloader
    """

    def __init__(self, questions, labels, q_position):
        self.questions = questions
        self.labels = labels
        self.q_position = q_position


    def __getitem__(self, item):
        return self.questions[item], self.labels[item], self.q_position[item]

    def __len__(self):
        return len(self.questions)

