# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2021/12/11 21:51
# @Function:

import dgl
'''
Given a graph (directed, edges from small node id to large):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5
# g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (3, 1), (2, 4), (3, 5)])
'''

# (4, 4),
g = dgl.DGLGraph([(0, 6), (1, 6), (2, 0), (3, 1), (5, 3), (6, 3), (7, 4), (8, 0)])
print(list(dgl.topological_nodes_generator(g)))