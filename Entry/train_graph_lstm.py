# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/12/4 17:07
# @Function:
import os
from DataProcess.Pipline import DataPipline
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from Config.ConfigT import MyConf
from Models.Graph_LSTM import GraphLSTM
import numpy as np
from gensim.models.word2vec import Word2Vec
from DataProcess.remove_cycle_edges_by_dfs import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import dgl
import pickle
import time
import torch


class Entry:
    def __init__(self, config):
        self.config = config
        self.root=self.config.data_path


    def load_data(self,part):
        '''
        加载数据集
        :param data_file: 数据集名称
        :return:
        '''
        graph_data_path=os.path.join(self.root, part,'graph_data_no_loop.pkl')
        if os.path.exists(graph_data_path):
            with open(graph_data_path,'rb') as f:
                data_lst = pickle.load(f)
            return data_lst
        else:
            data_lst=[]
            data_path = os.path.join(self.root, part,'blocks.pkl')
            with open(data_path,'rb') as f:
                dataset = pickle.load(f)

            for i in range(len(dataset)):
                if i%300==0:
                    print(i)
                g_dgl = dgl.DGLGraph()
                inst =  dataset.iloc[i]
                features = inst['graph_features']
                adjacency_list = inst['graph_adjacency']
                # 移除自循环的边
                new_adjacency_list, _ = dfs_remove_back_edges(adjacency_list)

                if len(inst['graph_features'])==0:
                    continue
                u = torch.tensor([x[0] for x in new_adjacency_list]) # 子节点id
                v = torch.tensor([x[1] for x in new_adjacency_list]) # 父节点id
                # 构造图
                g_dgl.add_edges(u, v)
                g_dgl.ndata['features'] = torch.tensor(features)
                label = int(inst['label'])
                data_lst.append((g_dgl, label))

            with open(graph_data_path,'wb') as f:
                pickle.dump(data_lst,f)

        return data_lst


    def collate(self, samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = [label-1 for label in labels] # 更新标签，分类的标签从0开始技术，而不是从1
        return batched_graph, torch.tensor(labels)

    def load_train_dev_test(self):
        # Use PyTorch's DataLoader and the collate function
        # defined before.
        train_loader = DataLoader(self.load_data('train'), batch_size = self.config.batch_size, shuffle=True, collate_fn = self.collate)
        dev_loader = DataLoader(self.load_data('dev'), batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
        test_loader = DataLoader(self.load_data('test'), batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
        return train_loader, dev_loader, test_loader

    def model_para_num(self, model):
        nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The number of paramete is %s' % nums)
        return nums

    def main(self):
        train_loader, dev_loader, test_loader = self.load_train_dev_test()
        # Create model
        model = GraphLSTM(self.config)
        # model = TreeGCN(self.config)
        self.model_para_num(model)
        if self.config.use_gpu:
            model.cuda()
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        model.train()

        train_loss_ = []
        test_loss_ = []
        train_result_ = []
        test_result_ = []

        for epoch in range(config.epochs):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            for train_iter, (train_bg, train_labels) in enumerate(train_loader):

                optimizer.zero_grad()
                prediction = model(train_bg)
                if config.use_gpu:
                    train_labels = train_labels.cuda()
                loss = loss_func(prediction, train_labels)
                loss.backward()
                if torch.isnan(loss):
                    print('wrong')

                optimizer.step()
                # print('Iter is %s, Loss value is %s' % (train_iter, loss.item()))
                # calc training acc
                _, predicted = torch.max(prediction.data, 1)
                total_acc += (predicted == train_labels).sum()
                total += len(train_labels)
                # total_loss += loss.item() * len(train_labels)
                total_loss += loss.item() * len(train_labels)
            train_loss_.append(total_loss / total)
            train_result_.append(total_acc.item() / total)

            total_loss = 0.0
            for test_iter, (test_bg, test_labels) in enumerate(test_loader):
                if config.use_gpu:
                    test_labels =  test_labels.cuda()
                prediction = model(test_bg)
                loss = loss_func(prediction, test_labels)
                _, predicted = torch.max(prediction.data, 1)
                total_acc += (predicted == test_labels).sum()
                total += len(test_labels)
                total_loss += loss.item() * len(test_labels)

            test_loss_.append(total_loss / total)
            test_result_.append(total_acc.item() / total)
            end_time = time.time()

            print('[Epoch: %3d/%3d] Train Loss: %.4f, Val Loss: %.4f,'
                  ' Train result: %s, Test result: %s, Time Cost: %.3f s'
                  % (epoch + 1, config.epochs, train_loss_[epoch], test_loss_[epoch],
                     train_result_[epoch], test_result_[epoch], end_time - start_time))

if __name__ == '__main__':
    config = MyConf('../Config/config.cfg')
    embedding_path = os.path.join(os.path.dirname(config.data_path), config.language, 'embedding')
    node_word2vec = Word2Vec.load(embedding_path + "/node_w2v_128").wv

    config.embeddings = np.zeros((node_word2vec.vectors.shape[0] + 1, node_word2vec.vectors.shape[1]), dtype="float32")
    config.embeddings[:node_word2vec.vectors.shape[0]] = node_word2vec.vectors
    config.embedding_dim = node_word2vec.vectors.shape[1]
    config.vocab = node_word2vec.vocab
    config.vocab_size = node_word2vec.vectors.shape[0] + 1

    entry = Entry(config)
    entry.main()

