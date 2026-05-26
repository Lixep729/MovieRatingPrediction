import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import Parameter

class MF(nn.Cell):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = Parameter(ms.Tensor([0,0], ms.float32), name="global_bias")
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        dot = self.reduce_sum(u * v, 1)
        u_b = self.user_bias(user_ids).squeeze()
        i_b = self.item_bias(item_ids).squeeze()
        return dot + u_b + i_b + self.global_bias

class NCF(nn.Cell):
    def __init__(self, num_users, num_items, layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        
        embed_dim = layers[0] // 2
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        mlp = []
        for i in range(len(layers)-1):
            mlp.append(nn.Dense(layers[i], layers[i+1]))
            mlp.append(nn.ReLU())
        mlp.append(nn.Dense(layers[-1], 1))
        self.mlp = nn.SequentialCell(mlp)

    def construct(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        concat = ops.Concat(axis=1)((u, v))
        out = self.mlp(concat)
        return out.squeeze()