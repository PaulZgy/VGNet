import torch
import torch.nn.functional as F
from torch.nn import Linear
from .base_mdel import BaseModel
from torch_geometric.nn import (SAGEConv, EdgePooling, global_mean_pool,
                                JumpingKnowledge)


class SageEdge(BaseModel):
    def __init__(self, name, num_layers=1, hidden=256):
        super(SageEdge, self).__init__(name)
        self.conv1 = SAGEConv(16, hidden)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            SAGEConv(hidden, hidden)
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 18)

    def reset_parameters(self):

        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data[4], data[3], data[7]
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        feature = x
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), feature

    def __repr__(self):
        return self.__class__.__name__
