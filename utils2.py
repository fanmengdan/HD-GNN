# -*- coding: utf-8 -*-
import numpy as np
# import pickle as pkl
from numpy import float32
import time
from tqdm import *
import os
import joblib

# 存储data，直接load 存储的 data
def read_data(self, step):

    repo = self.Repo
    # 把最耗时的部分储存起来
    dir = r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl'

    if not os.path.exists(dir):

        diagnal_is0_x = np.ones((self.Ne, self.Ne)) - np.eye(self.Ne)  #(200,200) 全1矩阵-单位阵
        diagnal_is0_y = np.ones((self.Nc, self.Nc)) - np.eye(self.Nc)  #(74,74) 全1矩阵-单位阵

        x = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)  # x是entity adj (100,200,200)
        y = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)  # y是hunk adj (100,74,74)
        IndexPathList = open(r'./dataset/'+repo+'/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')
        HunkIDmaps_path = open(r'./dataset/'+repo+'/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')
        IndexPaths = joblib.load(IndexPathList)
        HunkIDmaps = joblib.load(HunkIDmaps_path)

        node_x = np.zeros((x.shape[0], x.shape[1]))  # (100,200) 的零矩阵，100是batch大小

        bar1 = trange(x.shape[0]) # batchsize=100
        for i in bar1:  # 遍历100个图
            time.sleep(0.01)
            bar1.set_description('node_x[%i]' % i)  # 进度条左边显示信息
            node_x[i] = sum(x[i] * np.eye(self.Ne))  # x中第i个图 点乘取对角线矩阵，然后按行求和得到图x的各个节点的属性值
        node_x = node_x.reshape(x.shape[0], 1, self.Ne)  # (100,1,200)

        x_data = np.zeros((x.shape[0], x.shape[1], x.shape[1]))  # (100,200,200), 用来存储去掉对角线的adj
        y_label = np.zeros((y.shape[0], y.shape[1], y.shape[1]))  # (100,74,74)

        bar2 = trange(x.shape[0])
        for i in bar2:  # 遍历batch中每个图，i是batch中第i个图
            time.sleep(0.01)
            bar2.set_description('x_data[i,:,:] and y_label[i,:,:]')
            # x是数据，y是label
            x_data[i, :, :] = x[i, :, :] * diagnal_is0_x  # diagnal is removed，[:,:]是图的矩阵表示，点乘（全1矩阵-单位阵）
            y_label[i, :, :] = y[i, :, :] * diagnal_is0_y  # 去掉对角线的节点的属性值，只剩下边矩阵。i固定了，可以看到矩阵的原貌

        # entity relation 的源端/目的端entity、存在的边
        Es_data = np.zeros((100, self.Ne, self.Ner), dtype=float32)  # (100, 200, 39800) 每个entity节点对的 源端entity. batch_num, The Number of objects, The Number of Relations
        Et_data = np.zeros((100, self.Ne, self.Ner), dtype=float32)  # (100, 200, 39800) 每个entity节点对的 目的端entity.
        E_edge = np.zeros((100, self.Dr, self.Ner), dtype=float32)  # (100,2,39800) batch_num, The Relationship Dimension, The Number of Relations

        # Hunk relation 图 的源端/目的端hunk、存在的边
        Cs_label = np.zeros((100, self.Nc, self.Ncr), dtype=float32)  # (100, 74, 5402) 每个hunk节点对的 源端hunk.
        Ct_label = np.zeros((100, self.Nc, self.Ncr), dtype=float32)  # (100, 74, 5402) 每个hunk节点对的 目的端hunk.
        C_edge = np.zeros((100, self.Dr, self.Ncr), dtype=float32)  # (100,2,5402) batch_num, The Relationship Dimension, The Number of Relations

        # entity relation 的 源端/目的端hunk
        Esc_data = np.zeros((100, self.Nc, self.Ner), dtype=float32)  # (100,74,39800) 每个entity节点对的 源端hunk
        Etc_data = np.zeros((100, self.Nc, self.Ner), dtype=float32)  # (100,74,39800) 每个entity节点对的 目的端hunk

        # entity 图 的源端、目的端、每条边
        cnt = 0  # 遍历 number of relations
        # Relation Matrics R=<Rr,Rs,Ra>
        # 为每个对节点分配源端 存入Es_data、目的端 存入Et_data
        # 记录哪些节点对之间有边 E_edge
        bar3 = trange(self.Ne)
        for i in bar3:
            time.sleep(0.01)
            bar3.set_description('i in entity')
            for j in range(self.Ne):  # j 从 i+1 开始计算
                if (i != j):  # 每个图k的源端点和目的端点 Es_data 和 Et_data 都一样
                    # 每个entity节点对的 源端/目的端 entity
                    Es_data[:, i, cnt] = 1.0  # 200是源端点i (100,200,39800)  batch_num, The Number of Objects, The Number of Relations
                    Et_data[:, j, cnt] = 1.0  # 200是目的端点j (100,200,39800)  batch_num, The Number of Objects, The Number of Relations
                    for k in range(x_data.shape[0]):  # 遍历batchsize
                        # int(data[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                        # E_edge (取自x) != C_edge (取自y)。
                        # (100,2,39800) batch_num, The Relationship Dimension, The Number of Relations
                        # 只有存在边的节点对，第2个维度的第 int(x_data[k, i, j]) 个位置(0或者1) 上才是1
                        E_edge[k, int(x_data[k, i, j]), cnt] = 1
                    cnt += 1 # 39800 中的维度

            # hunk(label) 图 的源端、目的端、每条边
            cnt1 = 0  # 遍历 number of relations
            # Relation Matrics R=<Rr,Rs,Ra>
            # 为每个对节点分配源端 存入Cs_label、目的端 存入Ct_label
            # 记录哪些节点对之间有边 C_edge
            bar6 = trange(self.Nc)
            for i in bar6:
                time.sleep(0.01)
                bar6.set_description('i in hunk')
                for j in range(self.Nc):  # j 从 i+1 开始计算
                    if (i != j):  # 每个图k的源端点和目的端点 Es_data 和 Et_data 都一样
                        # 每个hunk节点对的 源端/目的端 hunk
                        Cs_label[:, i, cnt1] = 1.0  # 74是源端点i (100, 74, 5402)  batch_num, The Number of Objects, The Number of Relations
                        Ct_label[:, j, cnt1] = 1.0  # 74是目的端点j (100, 74, 5402)  batch_num, The Number of Objects, The Number of Relations
                        for k in range(y_label.shape[0]):  # 遍历batchsize
                            # int(y_label[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                            # E_edge (取自x) != C_edge (取自y)。
                            # (100,2,5402) batch_num, The Relationship Dimension, The Number of Relations
                            # 只有存在边的节点对，第2个维度的第int(y_label[k, i, j]个位置(0或者1) 上才是1
                            # y_label 是adj，其中有边就是1，没边就是0
                            C_edge[k, int(y_label[k, i, j]), cnt1] = 1
                        cnt1 += 1 # 5402 中的维度

        # 这段代码的目的是从 IndexPath 指向的文件中读取entity 索引的 hunk的 复杂id，
        # 然后根据这些索引在 HunkIDmap 中查找复杂id对应的 简单number，
        # 为每个 entity relation 匹配 源端hunk，存入 Esc_data 和 目的端hunk，存入 Etc_data
        bar8 = trange(x.shape[0])
        for index1 in bar8:  # 遍历100个图，每个图都有一个map
           time.sleep(0.01)
           bar8.set_description('IndexPath and HunkIDmap')
           # 识别i和j分别在哪个hunk里！！！
           IndexPath = IndexPaths[index1]
           HunkIDmap = HunkIDmaps[index1]  # 是一个dict

           # 读取index
           index_txt = open(IndexPath)
           indexLines = index_txt.readlines()[:self.Ne]  # 按行读取，限制entity 标号大小
           # 分配源端、目的端 hunkIDnum
           cnt2 = 0
           for i in range(len(indexLines)):  # 有的entity图可能比(200,200)小，也有可能大
               for j in range(len(indexLines)):
                   if (i != j):
                       Cs_key = indexLines[i].strip()
                       Ct_key = indexLines[j].strip()
                       if (Cs_key != 'null'):
                           Cs_num = HunkIDmap[Cs_key]
                           if (Cs_num < self.Nc):  # 限制hunkID 标号大小，因为cutting过
                               Esc_data[index1, Cs_num, cnt2] = 1.0
                       if (Ct_key != 'null'):
                           Ct_num = HunkIDmap[Ct_key]
                           if (Ct_num < self.Nc):
                               Etc_data[index1, Ct_num, cnt2] = 1.0
                       cnt2 += 1

        # x中每个图的节点特征, 当总样本数是奇数的时候不可以除以2
        E_node_train = node_x[0:int(x.shape[0] / 2)]  # (50,1,200)
        E_node_test = node_x[int(x.shape[0] / 2):x.shape[0]]  # (50,1,200)

        # x中每个图的边类型
        E_edge_train = E_edge[0:int(x.shape[0] / 2)]  # (50,2,39800)
        E_edge_test = E_edge[int(x.shape[0] / 2):x.shape[0]]  # (50,2,39800)

        # y中每个图的边类型
        C_edge_train = C_edge[0:int(x.shape[0] / 2)]  # (50,2,5402)
        C_edge_test = C_edge[int(x.shape[0] / 2):x.shape[0]]  # (50,2,5402)


        # save Esc_data、Etc_data
        with open(r'./Intermediate_products/'+repo+'/E_node_train_' + str(step) + '.pkl', 'wb') as f21:
            joblib.dump(E_node_train, f21)

        with open(r'./Intermediate_products/'+repo+'/E_node_test_' + str(step) + '.pkl', 'wb') as f22:
            joblib.dump(E_node_test, f22)

        with open(r'./Intermediate_products/'+repo+'/E_edge_train_' + str(step) + '.pkl', 'wb') as f23:
            joblib.dump(E_edge_train, f23)

        with open(r'./Intermediate_products/'+repo+'/E_edge_test_' + str(step) + '.pkl', 'wb') as f24:
            joblib.dump(E_edge_test, f24)

        with open(r'./Intermediate_products/'+repo+'/C_edge_train_' + str(step) + '.pkl', 'wb') as f25:
            joblib.dump(C_edge_train, f25)

        with open(r'./Intermediate_products/'+repo+'/C_edge_test_' + str(step) + '.pkl', 'wb') as f26:
            joblib.dump(C_edge_test, f26)

        with open(r'./Intermediate_products/'+repo+'/Es_data_' + str(step) + '.pkl', 'wb') as f27:
            joblib.dump(Es_data, f27)
            # pkl.dump(Esc_data, f27)

        with open(r'./Intermediate_products/'+repo+'/Et_data_' + str(step) + '.pkl', 'wb') as f28:
            joblib.dump(Et_data, f28)
            # pkl.dump(Etc_data, f28)

        with open(r'./Intermediate_products/'+repo+'/Cs_label_' + str(step) + '.pkl', 'wb') as f29:
            joblib.dump(Cs_label, f29)

        with open(r'./Intermediate_products/'+repo+'/Ct_label_' + str(step) + '.pkl', 'wb') as f30:
            joblib.dump(Ct_label, f30)

        with open(r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl', 'wb') as f31:
           joblib.dump(Esc_data, f31)
           # pkl.dump(Esc_data, f31)

        with open(r'./Intermediate_products/'+repo+'/Etc_data_' + str(step) + '.pkl', 'wb') as f32:
           joblib.dump(Etc_data, f32)
           # pkl.dump(Etc_data, f32)

    else:
        # pkl.load(open(r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl', 'rb'))
        # Etc_data = pkl.load(open(r'./Intermediate_products/'+repo+'/Etc_data_' + str(step) + '.pkl', 'rb'))

        # (50, 1, 200) ：
        # train/test 分别50个commit的entity-reference graph,
        # 每个graph的200个节点的属性，
        # 每个属性是1维数值
        # Entity node (E_node)
        E_node_train = joblib.load(open(r'./Intermediate_products/'+repo+'/E_node_train_' + str(step) + '.pkl', 'rb')) # (50, 1, 200)
        E_node_test = joblib.load(open(r'./Intermediate_products/'+repo+'/E_node_test_' + str(step) + '.pkl', 'rb')) # (50, 1, 200)

        # (50, 2, 39800)：200*200-200=39800
        # train/test 分别50个commit的 input entity-reference graph,
        # 每个graph的39800个节点对是否存在边，
        # 每个属性是2维数组[01]或[10]
        # Entity edge (E_edge)
        E_edge_train = joblib.load(open(r'./Intermediate_products/'+repo+'/E_edge_train_' + str(step) + '.pkl', 'rb')) # (50, 2, 39800)
        E_edge_test = joblib.load(open(r'./Intermediate_products/'+repo+'/E_edge_test_' + str(step) + '.pkl', 'rb')) # (50, 2, 39800)

        # (50, 2, 5402)：
        # train/test 分别50个commit的 output code change graph,
        # 每个graph的5402个节点对是否属于一组，
        # 每个属性是2维数组[01]或[10]
        # Code change edge (E_edge)
        C_edge_train = joblib.load(open(r'./Intermediate_products/'+repo+'/C_edge_train_' + str(step) + '.pkl', 'rb')) # (50, 2, 5402)
        C_edge_test = joblib.load(open(r'./Intermediate_products/'+repo+'/C_edge_test_' + str(step) + '.pkl', 'rb')) # (50, 2, 5402)

        # (100, 200, 39800):
        # 39800 个 entity-reference graphs 中
        # 每条边的 源端entity
        # Entity source (Es)
        Es_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Es_data_' + str(step) + '.pkl', 'rb')) # (100, 200, 39800)
        # 每条边的 目的端entity
        # Entity target (Et)
        Et_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Et_data_' + str(step) + '.pkl', 'rb')) # (100, 200, 39800)

        # (100, 74, 5402)：
        # 5402个code changes graphs 中
        # 每条边的 源端code change
        # Code change source (Cs)
        Cs_label = joblib.load(open(r'./Intermediate_products/'+repo+'/Cs_label_' + str(step) + '.pkl', 'rb'))
        # 每条边的 目的端code change
        # Code change target (Ct)
        Ct_label = joblib.load(open(r'./Intermediate_products/'+repo+'/Ct_label_' + str(step) + '.pkl', 'rb'))

        # (100, 74, 39800)：
        # 39800 个 entity-reference graphs 中
        # 每条边的 源端code change
        # Entity source code change(Esc)
        Esc_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl', 'rb'))
        # 每条边的 目的端code change
        # Entity target code change(Etc)
        Etc_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Etc_data_' + str(step) + '.pkl', 'rb'))

    return E_node_train,E_node_test,\
           E_edge_train,E_edge_test,\
           C_edge_train,C_edge_test,\
           Es_data,Et_data,\
           Cs_label,Ct_label,\
           Esc_data,Etc_data

