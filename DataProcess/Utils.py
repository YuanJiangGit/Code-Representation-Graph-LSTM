# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/12/7 19:53
# @Function:
import json
import pandas as pd
import os
import pickle

def tree_to_token_index(root_node):
    '''
    # tuple的list，tuple为((行，token的初始位置列),(行，结束位置))， 行和列都是从小到大的进行排序
    :param root_node:
    :return:
    '''
    if (len(root_node.children) == 0 or root_node.type == 'string_literal') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_node_index(root_node, seq):
    '''
    获得语法树中所有节点的token位置（jy）
    # tuple的list，tuple为((行，token的初始位置列),(行，结束位置))， 行和列都是从小到大的进行排序
    :param root_node:
    :return:
    '''
    seq.append(((root_node.start_point, root_node.end_point), root_node))
    for child in root_node.children:
        tree_to_node_index(child,seq)

def update_node_location(node_location,node_tokens):
    # node_location[0]是start_point (行, token偏移)
    token_len = len(node_tokens)
    node_location_str = str(node_location[0][0]+1)+'_'+str(node_location[0][1])+'_'+str(token_len) # 行_token行内偏移_token的长度
    return node_location_str

def c_node2ast(query_node_location, nodes_locat_dict):
    '''
    返回查询的PDG节点对应的语法子树
    :param query_node_location: e.g., # 4:1:15:17 (第4行，4行的第1个偏移位置，相对于整体程序的第15个偏移到第17个偏移)
    :param nodes_locat_dict:
    :return:
    '''

    # 4:1:15:17 改成 4:1:3 （）
    node_location_lst = [x for x in query_node_location.split(':')]
    query_node_location_str = node_location_lst[0] + '_' + node_location_lst[1] + '_' + str(int(node_location_lst[3]) - int(node_location_lst[2]) + 1)
    # 如果query_location在字典的key中，直接返回key对应的AST
    if query_node_location_str in nodes_locat_dict:
        return nodes_locat_dict[query_node_location_str]
    #
    q_line, q_start_point, q_len = [int(x) for x in query_node_location_str.split('_')]
    node_locat_lst = [(key, value) for key, value in nodes_locat_dict.items() if int(key.split('_')[0])==q_line]
    node_locat_lst.reverse() # 选择刚好小于query_node的AST，所以应该从line行的右往左比较

    allow_loss_len=0
    if 0<q_len<20:
        allow_loss_len = 2
    elif q_len>=20 and q_len<40:
        allow_loss_len = 3
    elif q_len>=40 and q_len < 50:
        allow_loss_len = 5
    elif q_len>=50 and q_len<70:
        allow_loss_len = 7
    elif q_len >= 70 and q_len < 90:
        allow_loss_len = 8
    elif q_len >= 90 and q_len < 120:
        allow_loss_len = 9
    elif q_len >= 120:
        allow_loss_len = 10

    for key, value in node_locat_lst:
        line, start_point, len =[int(x) for x in key.split('_')]
        if q_start_point>=start_point and q_len<=len+allow_loss_len: # 允许有一个单位的误差，因为statement的分号的缘故
            return value

    for key, value in node_locat_lst:
        line, start_point, len =[int(x) for x in key.split('_')]
        if q_start_point+2>=start_point and q_len<=len+allow_loss_len: # 允许有一个单位的误差，因为statement的分号的缘故
            return value

    # 如果实在解析不对，返回一个当前行的AST
    return node_locat_lst[-1][1]

def java_node2ast(query_node_location, nodes_locat_dict):
    '''
    返回查询的PDG节点对应的语法子树
    :param query_node_location: e.g., # 4:1:15:17 (第4行，4行的第1个偏移位置，相对于整体程序的第15个偏移到第17个偏移)
    :param nodes_locat_dict:
    :return:
    '''

    #query node location form: 4:1:3
    node_location_lst = [x for x in query_node_location.split(':')]
    query_node_location_str = str(int(float(node_location_lst[0])) )+ '_' + node_location_lst[1] + '_' + str(node_location_lst[2])
    # 如果query_location在字典的key中，直接返回key对应的AST
    if query_node_location_str in nodes_locat_dict:
        return nodes_locat_dict[query_node_location_str]
    #
    q_line, q_start_point, q_len = [int(x) for x in query_node_location_str.split('_')]
    node_locat_lst = [(key, value) for key, value in nodes_locat_dict.items() if int(key.split('_')[0])==q_line]
    node_locat_lst.reverse() # 选择刚好小于query_node的AST，所以应该从line行的右往左比较

    allow_loss_len=0
    if 0<q_len<20:
        allow_loss_len = 2
    elif q_len>=20 and q_len<40:
        allow_loss_len = 3
    elif q_len>=40 and q_len < 50:
        allow_loss_len = 5
    elif q_len>=50 and q_len<70:
        allow_loss_len = 7
    elif q_len >= 70 and q_len < 90:
        allow_loss_len = 8
    elif q_len >= 90 and q_len < 120:
        allow_loss_len = 9
    elif q_len >= 120:
        allow_loss_len = 10

    for key, value in node_locat_lst:
        line, start_point, len =[int(x) for x in key.split('_')]
        if q_start_point>=start_point and q_len<=len+allow_loss_len and q_len+allow_loss_len>=len: # 允许有一个单位的误差，因为statement的分号的缘故
            return value

    for key, value in node_locat_lst:
        line, start_point, len =[int(x) for x in key.split('_')]
        if q_start_point+2>=start_point and q_len<=len+allow_loss_len and q_len+allow_loss_len>=len: # 允许有一个单位的误差，
            return value

    distance = float('inf')
    nearest_node = node_locat_lst[-1][1]
    for key, value in node_locat_lst:
        line, start_point, len = [int(x) for x in key.split('_')]
        if q_start_point > start_point:
            continue
        ast_end_point = start_point+len
        q_end_point = q_start_point+q_len
        if abs(ast_end_point-q_end_point)+abs(start_point-q_start_point)<distance:
            nearest_node = value
            distance = abs(ast_end_point-q_end_point)+abs(start_point-q_start_point)
    return nearest_node

    # 如果实在解析不对，返回一个当前行的AST
    # return node_locat_lst[-1][1]


def tree_to_variable_index(root_node, index_to_code):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens


def index_to_code_token(index, code):
    '''
    根据code的位置得到对应的code
    '''
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def complete_id():
    dataset=pd.read_pickle('../resources/dataset/c/program_pdg.pickle')
    # dataset.columns=['code', 'pdg', 'label', 'file_name']
    with open('../resources/dataset/c/filename_id.json', 'r') as f:
        dict=json.load(f)

    data_path = os.path.join('../resources/dataset/c/programs.pkl')
    sources = pd.read_pickle(data_path)
    sources.columns = ['id', 'code', 'label']

    count=0
    def find_id(file_name):
        if file_name in dict.keys():
            return dict[file_name]
        else:
            print('wrong key : %s'% file_name)
            return None

    def find_source_code(id,sources):
        sources = sources[sources['id']==id]
        source_code = sources.iloc[0]['code']
        return source_code

    for i in range(len(dataset)):
        # (id, nodes_dict, edges_tuple, label, file_name)
        nodes_dict, edges_tuple, label, file_name = dataset[i]
        id = find_id(file_name)
        if id == None:
            id=-1
        if id!=-1:
            source_code = find_source_code(id, sources)
        else:
            with open(r'E:\PyProject\Code_representation\Code_data_augmentation\resources\OJ Benchmark\\'+file_name+'.c','r') as f:
                source_code = f.read()
        dataset[i] = (id, source_code, nodes_dict, edges_tuple, label, file_name)

    # dataset['orig_code']=dataset.apply(lambda x:find_source_code(x['id'], sources),axis=1)
    with open('../resources/dataset/c/program_code_pdg.pickle','wb') as f:
        pickle.dump(dataset, f)
    print(len(dataset))


def read_data():
    graph_data_path = os.path.join('../resources/dataset/c/train/graph_data.pkl')
    if os.path.exists(graph_data_path):
        with open(graph_data_path, 'rb') as f:
            data_lst = pickle.load(f)
        return data_lst

def merge_coj_comp():
    '''
    将OJ benchmark和OJ complement 数据集合并
    :return:
    '''
    dir_name=['c','c_exp']
    root='../resources/dataset/'
    data_name='All_data_input.pkl'
    path=os.path.join(root,'c',data_name)
    c_source = pd.read_pickle(path)
    c_source.columns = ['id','code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name', 'tree-graph']
    c_source.drop('id', axis=1, inplace = True)

    path=os.path.join(root,'c_exp',data_name)
    c_exp_source = pd.read_pickle(path)
    c_exp_source.columns = ['code_str', 'node_lst', 'node_location', 'pdg', 'label', 'file_name', 'tree-graph']
    c_all = pd.concat([c_source, c_exp_source])
    c_all.to_pickle(os.path.join(root,'c_all',data_name))


def merge_orig_code():
    dir_name = ['OJ-Data-1', 'OJ-Data-2']
    root = '../resources/dataset/'
    data_name = 'programs.pkl'
    path = os.path.join(root, dir_name[0], data_name)
    oj_1_source = pd.read_pickle(path)
    oj_1_source.columns = ['id', 'code', 'label']
    path = os.path.join(root, dir_name[1], data_name)
    oj_2_source = pd.read_pickle(path)

    c_all = pd.concat([oj_1_source, oj_2_source])
    c_all.to_pickle(os.path.join(root, 'OJ-All', data_name))

def read_merge_data():
    data_path = '../resources/dataset/OJ-All/programs.pkl'
    if os.path.exists(data_path):
        data = pd.read_pickle(data_path)
        for i in range(20):
            print(data.iloc[i])


if __name__ == '__main__':
    # complete_id()
    # read_data()
    # merge_orig_code()
    read_merge_data()