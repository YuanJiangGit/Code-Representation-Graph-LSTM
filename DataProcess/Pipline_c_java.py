# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/12/7 19:42
# @Function:
from DataProcess.Utils import *
from tree_sitter import Language, Parser
import pandas as pd
import pickle
import os
import nltk

class DataPipline:
    def __init__(self, ratio, language):
        self.ratio = ratio
        root = os.getcwd() + '/../resources/dataset/'
        self.root = os.path.join(root, language)
        self.language = language
        self.parser = self.load_parsers(language)
        self.count=0
    def load_parsers(self,lang):
        # load parsers
        LANGUAGE = Language(
            'F:\PyProject_two\Tree_lstm_dgl_classification\DataProcess\parser1\my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        return parser


    def gen_sing_tree(self, node, code):
        '''
        得到我们想要的树
        '''
        if (len(node.children) == 0 or node.type == 'string_literal') and node.type != 'comment':
            feature = index_to_code_token((node.start_point, node.end_point), code)  # 如果是叶子节点获得对应的token，否则获得类型
        else:
            feature = node.type
        childs = []

        location = (node.start_point, node.end_point)
        node_tokens = index_to_code_token(location, code)
        node_location_str = update_node_location(location, node_tokens)

        for child in node.children:
            childs.append(self.gen_sing_tree(child, code))
        return {"features": feature, "children": childs, "node_tokens": node_tokens, "node_location": node_location_str}

    def _label_node_index(self, node, n=0):
        '''
        给node加上id
        :param node:
        :param n:
        :return:
        '''
        node['index'] = n
        for child in node['children']:
            n = self._label_node_index(child, n + 1)
        return n

    def sing_tree2dict(self,single_tree, dict):
        '''
        将迭代的single_tree转化为dict形式
        :param single_tree:
        :param dict:
        :return:
        '''
        dict[single_tree['node_location']] = {"features": single_tree["features"], "node_tokens": single_tree["node_tokens"], "index": single_tree["index"], "children":single_tree["children"]}
        for child in single_tree['children']:
            self.sing_tree2dict(child, dict)

    def _gather_node_attributes(self, node, key):
        features = [node[key]]
        for child in node['children']:
            features.extend(self._gather_node_attributes(child, key))
        return features

    def _gather_adjacency_list(self,node):
        adjacency_list = []
        for child in node['children']:
            adjacency_list.append([child['index'], node['index']]) # 这里是孩子节点指向父节点
            adjacency_list.extend(self._gather_adjacency_list(child))

        return adjacency_list


    # parse source code
    def parse_source(self, output_file, option):
        path = self.root + '/' + output_file
        if os.path.exists(path) and option == 'existing':
            return 'Exist'
        else:
            with open(self.root + '/program_code_pdg.pickle', 'rb') as f:
                source = pickle.load(f, encoding='latin1')
            if self.language is 'c' or self.language is 'c_exp' or self.language is 'c_all':
                dataset = pd.DataFrame(columns=['id', 'code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name'])
                for i in range(len(source)):
                    print(i)
                    (id, origin_code, nodes_dict, edges_tuple, label, file_name) = source[i]
                    node_lst = []
                    node_location = []
                    for node in nodes_dict:  # nodes_dict is list, consisting of dicts
                        statement = node['code']
                        node_lst.append(statement)
                        node_location.append(node['location'])
                    edge_dict = {}
                    # self.columns = ['code', 'pdg', 'label', 'file_name']
                    # node_list 和 node_location是一一对应的，每一行对应一个节点
                    instance = {'id': id, 'code_str': origin_code, 'node_lst': node_lst, 'node_location':node_location, 'pdg': edges_tuple, 'label': label, 'file_name': file_name}
                    dataset = dataset.append(instance, ignore_index=True)
                dataset.to_pickle(path)
            else:
                dataset = pd.DataFrame(columns=['id', 'code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name'])
                for i in range(len(source)):
                    # print(i)
                    (origin_code, nodes_dict, edges_tuple, label, file_name) = source[i]
                    node_lst = []
                    node_location = []
                    for node in nodes_dict:  # nodes_dict is list, consisting of dicts
                        statement = node['code']
                        node_lst.append(statement)
                        node_location.append(node['location'])
                    edges_tuple_temp = []
                    for edge in edges_tuple:  # edges_tuple is list, consisting of tuples
                        start = edge[0]
                        end = edge[1]
                        edge_label = edge[2]
                        if end - start != 1:
                            edges_tuple_temp.append((start, end, edge_label))
                    # self.columns = ['code', 'pdg', 'label', 'file_name']
                    instance = {'id': label, 'code_str': origin_code, 'node_lst': node_lst, 'node_location':node_location, 'pdg': edges_tuple_temp, 'label': label, 'file_name': file_name}
                    dataset = dataset.append(instance, ignore_index=True)
                dataset.to_pickle(path)


    def update_dataset(self, dataset,file_name):
        path = os.path.join(r'E:\PyProject\Code_representation\Code_data_augmentation\resources\OJ Benchmark\\'+file_name+'.c')
        with open(path,'r') as f:
            code_str = f.read()
        data_path = os.path.join(self.root, dataset)
        sources = pd.read_pickle(data_path)
        sources.columns = ['id', 'code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name']
        sources.loc[sources['file_name'] == file_name,'code_str']=code_str
        # inst.iloc[0]['code_str']=code_str
        sources.to_pickle(os.path.splitext(data_path)[0]+'1.pkl')

    def extract_AST(self, code_str,node_lst,node_locations, pdg):
        '''
        提取code对应的语法树AST
        :param parser:
        :param code:
        :return:
        '''
        # print(file_name)
        self.count+=1
        print(self.count)
        tree = self.parser.parse(bytes(code_str, 'utf8'))
        root_node = tree.root_node

        code = code_str.split('\n')
        sing_tree = self.gen_sing_tree(root_node, code)

        self._label_node_index(sing_tree, 0)
        # 按照顺序语法树节点对应的feature(e.g.,token)
        features = self._gather_node_attributes(sing_tree, 'features')

        # PDG+AST节点之间关系
        adjacency_list = [[start, end] for (start, end, edge_label) in pdg]

        nodes_locat_dict={}
        self.sing_tree2dict(sing_tree, nodes_locat_dict)

        node_id_count = len(node_lst) # node_id 计数
        # AST+PDG图对应的edge
        new_graph_features = ['' for i in range(node_id_count)]
        for index, node in enumerate(node_lst):  # index代表第几个pdg_nodes
            node_tokens = node_lst[index]
            # print('-------------')
            # print('Original Tokens %s'% node_tokens)
            # 4:1:15:17
            node_location = node_locations[index]
            if self.language in ['c','c_exp', 'c_all']:
                # find the corresponding sub-ast tree of node
                node_ast_info =c_node2ast(node_location, nodes_locat_dict)
            else:
                node_ast_info = java_node2ast(node_location, nodes_locat_dict)

            # PDG node对应的feature
            new_graph_features[index]=node_ast_info['features']
            # print('Find Ast Info:')
            # relabel index for sub-tree, (sub-tree root node id is node_id_count)
            self._label_node_index(node_ast_info, node_id_count)
            node_ast_feature = self._gather_node_attributes(node_ast_info,'features') # 类似于 ['binary_expression', 'subscript_expression', 'mul_save', '[', 'len', ']', '!=', '0']

            # pdg node --> sub-ast tree root node
            adjacency_list.append([index, node_id_count])
            # sub-ast tree adjacency list
            adjacency_list.extend(self._gather_adjacency_list(node_ast_info)) #
            new_graph_features.extend(node_ast_feature)

            node_id_count += len(node_ast_feature)

        return {'tree-features': features, 'graph_adjacency_list': adjacency_list, 'graph_features':new_graph_features}


    def gen_data_tree(self, input_file, output_file, option):
        '''
        对每个数据实例，生成符合Tree-LSTM模型的输入形式
        '''
        store_path=os.path.join(self.root, output_file)
        if os.path.exists(store_path) and option == 'existing':
            sources = pd.read_pickle(store_path)
            # 以下用于测试，只保留label在[1，10]
            # sources = sources[sources['label'].isin(range(1, 10))]
        else:
            data_path = os.path.join(self.root, input_file)
            sources = pd.read_pickle(data_path)
            sources.columns = ['id', 'code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name']
            sources['tree-graph'] = sources.apply(lambda x: self.extract_AST(x['code_str'],x['node_lst'],x['node_location'],x['pdg']), axis=1)
            sources.to_pickle(store_path)
        self.sources=sources
        return sources

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root+ '/'
        # if self.sources == None:
        #     self.sources = pd.read_pickle(self.root + self.language + '/' + dataset_name)
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)  # random_state=666
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = os.path.join(self.root, 'train', 'train_.pkl')
        data = pd.read_pickle(input_file)
        embedding_path = os.path.join(self.root, 'embedding')
        if not os.path.exists(embedding_path):
            os.mkdir(embedding_path)

        def trans_to_sequences(ast_info):
            return ast_info['tree-features']

        corpus = data['tree-graph'].apply(trans_to_sequences)
        # for index in range(len(trees)):
        #     print(index)
        #     print(trans_to_sequences(trees['code'].iloc[index]))

        str_corpus = [' '.join(c) for c in corpus]
        data['code'] = pd.Series(str_corpus)
        data.to_csv(embedding_path + '/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(embedding_path + '/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self, dataset_name, size):
        from gensim.models.word2vec import Word2Vec

        path = pd.read_pickle(self.root + '/' + dataset_name + '/' + dataset_name + '_.pkl')
        embedding_path = os.path.join(self.root, 'embedding')
        word2vec = Word2Vec.load(embedding_path + '/node_w2v_' + str(size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def trans_ast_to_blocks(input):
            features = input['tree-features']
            feature_token_index = [vocab[token].index if token in vocab else max_token for token in features]
            input['tree-features'] = feature_token_index
            return input

        def trans_token_to_blocks(input):
            graph_features = input['graph_features']
            sequence = [vocab[token].index if token in vocab else max_token for token in graph_features]
            return sequence

        def trans_edge_label(input):
            graph_adjacency_list = input['graph_adjacency_list']
            return graph_adjacency_list


        data = pd.DataFrame(path, copy=True)
        data['graph_features'] = data.apply(lambda x: trans_token_to_blocks(x['tree-graph']), axis=1)
        data['graph_adjacency'] = data.apply(lambda x: trans_edge_label(x['tree-graph']), axis=1)
        # if 'label' in data.columns:
        #     data.drop('label', axis=1, inplace=True)
        self.blocks = data
        data.to_pickle(os.path.join(self.root, dataset_name) + '/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        dataset_name = 'All_data_1.pkl'
        # self.parse_source(output_file=dataset_name, option='existing')  # option='existing'
        dataset_input = 'All_data_input.pkl'
        self.gen_data_tree(input_file=dataset_name, output_file=dataset_input, option='existing')  # option='existing'
        print('split data...')
        # self.split_data()
        print('train word embedding...')
        # size = 128
        # self.dictionary_and_embedding(None, size)
        print('generate block sequences...')
        # self.generate_block_seqs('train', size)
        # self.generate_block_seqs('dev', size)
        # self.generate_block_seqs('test', size)

    def tst_ast(self):
        dir='./parser1/c_example/'
        parser = self.load_parsers('c')
        for file in os.listdir(dir):
            file_name = os.path.join(dir, file)
            with open(file_name,'r') as f:
                code = f.read()
            self.extract_AST(code)
if __name__ == '__main__':
    # c, c_exp, c_all
    pipline = DataPipline('3:1:1', 'java')
    pipline.run()
    # pipline.update_dataset('All_data_.pkl','86-86')