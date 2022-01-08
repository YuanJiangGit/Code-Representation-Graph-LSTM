import os
import pickle
import dgl

def train_topo_check():
    graph_data_path = os.path.join('../resources/dataset/c/test/graph_data_no_loop.pkl')
    with open(graph_data_path, 'rb') as f:
        data_lst = pickle.load(f)
    print('The length of original dataset is %s'%len(data_lst))

    topolog_len_lst = {}
    for i in range(len(data_lst)):
        inst =data_lst[i]
        # print(inst)
        topolog = list(dgl.topological_nodes_generator(inst[0]))
        topolog_len_lst[i]=(len(topolog), inst)
        # print(len(topolog))
    l = sorted(topolog_len_lst.items(), key = lambda kv:(kv[1][0], kv[0]))
    print(l)
    for key, value in l:
        if value[0]>90: # 拓扑结构深度超过90的删除掉
            data_lst.remove(value[1])
    print('The length of modified dataset is %s' % len(data_lst))

    with open(graph_data_path, 'wb') as f:
        pickle.dump(data_lst, f)



def train_check():
    graph_data_path = os.path.join('../resources/dataset/c-backup/train/graph_data_no_loop.pkl')
    with open(graph_data_path, 'rb') as f:
        data_lst = pickle.load(f)
    # print(len(data_lst))

    block_data_path = os.path.join(os.path.dirname(graph_data_path),'blocks.pkl')
    with open(block_data_path, 'rb') as f:
        dataset = pickle.load(f)
    # print(len(dataset))

    # for i in range(len(data_lst)):

    wrong_inst=data_lst[23620]
    wrong_inst_orig = dataset.iloc[23621]
    print(wrong_inst)
    print(wrong_inst_orig)
    topolog = list(dgl.topological_nodes_generator(wrong_inst[0]))
    print(topolog)
    print(len(topolog))

def filter_data():
    graph_data_path = os.path.join('../resources/dataset/c/train/graph_data_no_loop.pkl')
    with open(graph_data_path, 'rb') as f:
        data_lst = pickle.load(f)

    print(data_lst)
    del data_lst[23620]
    print(data_lst)

    with open(graph_data_path, 'wb') as f:
        pickle.dump(data_lst, f)



if __name__ == '__main__':
    # train_check()
    # filter_data()
    train_topo_check()