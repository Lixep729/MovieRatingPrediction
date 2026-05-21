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
        self.global_bias = Parameter(ms.Tensor([3.5], ms.float32), name="global_bias")
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        dot = self.reduce_sum(u * v, 1)
        u_b = self.user_bias(user_ids).squeeze()
        i_b = self.item_bias(item_ids).squeeze()
        return dot + u_b + i_b + self.global_bias

class NCF(nn.Cell):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        fc = []
        for i in range(len(layers)-1):
            fc.append(nn.Dense(layers[i], layers[i+1]))
            if i != len(layers)-2:
                fc.append(nn.ReLU())
        self.fc_layers = nn.SequentialCell(fc)
        self.output_layer = nn.Dense(layers[-1], 1)

    def construct(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        x = ops.concat([u_emb, i_emb], axis=1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x.squeeze(1)