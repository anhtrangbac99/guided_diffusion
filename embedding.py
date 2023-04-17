import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        # self.condEmbedding = nn.Sequential(
        #     # nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
        #     nn.Conv2d(3,1,kernel_size=1),
        #     nn.Linear(32, dim),
        #     nn.SiLU(),
        #     nn.Linear(dim, dim),
        #     # nn.Conv2d(dim,dim,kernel_size=1)

        # )

        # self.condEmbedding = nn.Sequential(
        #     nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
        #     nn.Linear(d_model, dim),
        #     nn.SiLU(),
        #     nn.Linear(dim, dim),
        # )
        self.conv1 = nn.Conv2d(3,1,kernel_size=1)
        # self.pool1 = nn.MaxPool2d(3)
        self.SiLU = nn.SiLU()
        self.conv2 = nn.Conv2d(1,1,kernel_size=1)
        # self.pool2 = nn.MaxPool2d(3)
        self.linear = nn.Linear(128,dim)
        self.linear2 = nn.Linear(128*dim,256)

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        # emb = self.condEmbedding(t)
        emb = self.conv1(t)
        # print(emb.shape)
        # emb = self.pool1(emb)
        emb = self.SiLU(emb)
        # print(emb.shape)
        emb = self.conv2(emb)
        # emb = self.pool2(emb)
        # print(emb.shape)

        emb = self.linear(emb)
        emb = emb.view(t.shape[0],-1)
        emb = self.linear2(emb)
        return emb


# class ConditionalEmbedding(nn.Module):
#     def __init__(self, num_labels:int, d_model:int, dim:int):
#         assert d_model % 2 == 0
#         super().__init__()
#         self.condEmbedding = nn.Sequential(
#             nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
#             nn.Linear(d_model, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim),
#         )

#     def forward(self, t:torch.Tensor) -> torch.Tensor:
#         emb = self.condEmbedding(t)
#         return emb
