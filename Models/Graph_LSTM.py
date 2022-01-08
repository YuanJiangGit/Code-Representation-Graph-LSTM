# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/12/7 19:40
# @Function:


import torch as th
import torch.nn as nn
import dgl.function as fn
import dgl

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'] #
        # 求和
        h_head =  h_cat.sum(1) # dim: h_size torch.Size([256])

        # equation (2)
        f = th.sigmoid(self.U_f(h_cat)) #
        # second term of equation (5)
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_head)+nodes.data['iou'], 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou # [64*768]
        i, o, u = th.chunk(iou, 3, 1) # [64,256]
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c'] # [64,256]
        # equation (6)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}


class GraphLSTM(nn.Module):
    def __init__(self,config):
        super(GraphLSTM, self).__init__()
        # num_vocabs,
        #                  x_size,
        #                  h_size,
        #                  num_classes,
        #                  dropout,
        #                  pretrained_emb=None
        # 1, 256, 3
        self.hidden_dim = config.hidden_dim
        n_classes = config.class_num
        self.th = th.cuda if config.use_gpu else th
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        embedding_dim = config.embedding_dim
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.x_size = embedding_dim
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.hidden_dim, n_classes)
        self.cell = TreeLSTMCell(self.x_size, self.hidden_dim)

        # pretrained  embedding
        if config.embeddings is not None:
            self.embedding.weight.data.copy_(th.from_numpy(config.embeddings))
            self.embedding.weight.requires_grad = True

    def forward(self, batch):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """

        g = batch.to(self.device)
        # to heterogenous graph
        n = g.number_of_nodes()
        h = th.zeros((n, self.hidden_dim)).to(self.device)
        c = th.zeros((n, self.hidden_dim)).to(self.device)

        features = g.ndata['features']
        # feed embedding
        embeds = self.embedding(self.th.LongTensor(features))
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds))
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        # h = self.dropout(g.ndata.pop('h'))
        hg = dgl.mean_nodes(g, 'h')
        logits = self.linear(hg)
        return logits