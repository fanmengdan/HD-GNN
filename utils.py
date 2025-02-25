# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from numpy import float32
import time
from tqdm import *
import os

# 没有存储data，实时处理并load
def read_data(self, step):
    diagnal_is0_x = np.ones((self.No, self.No)) - np.eye(self.No)  # 全1矩阵-单位阵
    diagnal_is0_y = np.ones((self.HNo, self.HNo)) - np.eye(self.HNo)  # 全1矩阵-单位阵

    x = np.load("./Adjset/Cutting_Adjs/CAdjs_" + str(step) + ".npy", allow_pickle=True)  # x是数据图 (100,200,200)
    y = np.load("./Adjset/Cutting_Adjs/CHunkAdjs_" + str(step) + ".npy", allow_pickle=True)  # y是label图 (100,74,74)
    IndexPatCt_path = open(r'./Adjset/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')
    HunkIDmaps_path = open(r'./Adjset/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')
    IndexPaths = pkl.load(IndexPatCt_path)
    HunkIDmaps = pkl.load(HunkIDmaps_path)

    node_x = np.zeros((x.shape[0], x.shape[1]))  # (100,200) 的零矩阵，100是batch大小

    bar1 = trange(x.shape[0])
    for i in bar1:  # 遍历100个图
        time.sleep(0.01)
        bar1.set_description("node_x[%i]" % i)  # 进度条左边显示信息
        node_x[i] = sum(x[i] * np.eye(self.No))  # x中第i个图 点乘取对角线矩阵，然后按行求和得到图x的各个节点的属性值
    node_x = node_x.reshape(x.shape[0], 1, self.No)  # (100,1,200)

    x_data = np.zeros((x.shape[0], x.shape[1], x.shape[1]))  # (100,200,200)
    y_label = np.zeros((y.shape[0], y.shape[1], y.shape[1]))  # (100,74,74)

    bar2 = trange(x.shape[0])
    for i in bar2:  # 遍历batch中每个图，i是batch中第i个图
        time.sleep(0.01)
        bar2.set_description("x_data[i,:,:] and y_label[i,:,:]")
        # x是数据，y是label
        x_data[i, :, :] = x[i, :, :] * diagnal_is0_x  # diagnal is removed，[:,:]是图的矩阵表示，点乘单位阵a
        y_label[i, :, :] = y[i, :, :] * diagnal_is0_y  # 去掉对角线的节点的属性值，只剩下边矩阵。i固定了，可以看到矩阵的原貌

    # entity 图 的源端、目的端、每条边
    Es_data = np.zeros((100, self.No, self.Nr), dtype=float32);  # (100,200,39800) 每个entity节点对的 源端entity
    Et_data = np.zeros((100, self.No, self.Nr), dtype=float32);  # (100,200,39800) 每个entity节点对的 目的端entity
    E_edge = np.zeros((100, self.Dr, self.Nr), dtype=float32);  # (100,2,39800) batch_num, The Relationship Dimension, The Number of Relations

    # Hunk 图
    Cs_label = np.zeros((100, self.HNo, self.HNr), dtype=float32);  # (100,74,5476) 每个hunk节点对的 源端hunk
    Ct_label = np.zeros((100, self.HNo, self.HNr), dtype=float32);  # (100,74,5476) 每个hunk节点对的 目的端hunk
    C_edge = np.zeros((100, self.Dr, self.HNr), dtype=float32);  # (100,2,5476) batch_num, The Relationship Dimension, The Number of Relations

    # 每条entity边 的 源端/目的端 hunk
    Esc_data = np.zeros((100, self.HNo, self.Nr), dtype=float32);  # (100,74,39800) 每个entity节点对的 源端hunk
    Etc_data = np.zeros((100, self.HNo, self.Nr), dtype=float32);  # (100,74,39800) 每个entity节点对的 目的端hunk

    # entity 图 的源端、目的端、每条边
    cnt = 0  # 遍历 number of relations
    # Relation Matrics R=<Rr,Rs,Ra>
    # 为每个relation 分配 源端i 和 目的端j
    bar3 = trange(self.No)
    for i in bar3:
        time.sleep(0.01)
        bar3.set_description("i in entity")
        for j in range(self.No):  # j 从 i+1 开始计算
            if (i != j):  # 每个图k的源端点和目的端点 Es_data 和 Et_data 都一样
                # 每个entity节点对的 源端/目的端 entity
                Es_data[:, i,
                cnt] = 1.0;  # 200是源端点i (100,200,39800)  batch_num, The Number of Objects, The Number of Relations
                Et_data[:, j,
                cnt] = 1.0;  # 200是目的端点j (100,200,39800)  batch_num, The Number of Objects, The Number of Relations
                for k in range(x_data.shape[0]):  # 遍历batch
                    # int(data[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                    # E_edge (取自x) != C_edge (取自y)。
                    E_edge[k, int(x_data[
                                       k, i, j]), cnt] = 1  # (100,2,39800) batch_num, The Relationship Dimension, The Number of Relations
                cnt += 1;

    # hunk 图 的源端、目的端、每条边
    cnt1 = 0  # 遍历 number of relations
    # Relation Matrics R=<Rr,Rs,Ra>
    # 为每个relation 分配 源端i 和 目的端j
    bar6 = trange(self.HNo)
    for i in bar6:
        time.sleep(0.01)
        bar6.set_description("i in hunk")
        for j in range(self.HNo):  # j 从 i+1 开始计算
            if (i != j):  # 每个图k的源端点和目的端点 Es_data 和 Et_data 都一样
                # 每个hunk节点对的 源端/目的端 hunk
                Cs_label[:, i, cnt1] = 1.0;  # 74是源端点i (100,74,5476)  batch_num, The Number of Objects, The Number of Relations
                Ct_label[:, j, cnt1] = 1.0;  # 74是目的端点j (100,74,5476)  batch_num, The Number of Objects, The Number of Relations
                for k in range(y_label.shape[0]):  # 遍历batch
                    # int(y_label[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                    # E_edge (取自x) != C_edge (取自y)。
                    C_edge[k, int(y_label[k, i, j]), cnt1] = 1  # (100,2,5476) batch_num, The Relationship Dimension, The Number of Relations
                cnt1 += 1

    # cutting 为每个relation 匹配 源端hunk 和 目的端hunk
    bar8 = trange(x.shape[0])
    for index1 in bar8:  # 遍历100个图，每个图都有一个map
       time.sleep(0.01)
       bar8.set_description("IndexPath and HunkIDmap")
       # 识别i和j分别在哪个hunk里
       IndexPath = IndexPaths[index1]
       HunkIDmap = HunkIDmaps[index1]  # 是一个dict

       # 读取index
       index_txt = open(IndexPath)
       indexLines = index_txt.readlines()[:200]  # 按行读取，限制entity 标号大小
       # 分配源端、目的端 hunkIDnum
       cnt2 = 0
       for i in range(len(indexLines)):  # 有的entity图可能比(200,200)小，也有可能大
           for j in range(len(indexLines)):
               if (i != j):
                   Cs_key = indexLines[i].strip()
                   Ct_key = indexLines[j].strip()
                   if (Cs_key != 'null'):
                       Cs_num = HunkIDmap[Cs_key]
                       if (Cs_num < 74):  # 限制hunkID 标号大小
                           Esc_data[index1, Cs_num, cnt2] = 1.0
                   if (Ct_key != 'null'):
                       Ct_num = HunkIDmap[Ct_key]
                       if (Ct_num < 74):
                           Etc_data[index1, Ct_num, cnt2] = 1.0
                   cnt2 += 1;


    # x中每个图的节点特征
    E_node_train = node_x[0:int(x.shape[0] / 2)]  # (50,1,200)
    E_node_test = node_x[int(x.shape[0] / 2):x.shape[0]]  # (50,1,200)

    # x中每个图的边的关系
    E_edge_train = E_edge[0:int(x.shape[0] / 2)]  # (50,2,39800)
    E_edge_test = E_edge[int(x.shape[0] / 2):x.shape[0]]  # (50,2,39800)

    # y中每个图的边的关系
    C_edge_train = C_edge[0:int(x.shape[0] / 2)]  # (50,2,5476)
    C_edge_test = C_edge[int(x.shape[0] / 2):x.shape[0]]  # (50,2,5476)

    return E_node_train,E_node_test,\
          E_edge_train,E_edge_test,\
          C_edge_train,C_edge_test,\
          Es_data,Et_data,\
          Cs_label,Ct_label,\
          Esc_data,Etc_data

