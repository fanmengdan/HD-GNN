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
    # 把最耗时的部分储存起来
    dir = r'./Intermediate_products/elasticsearch/RHr_data_' + str(step) + '.pkl'

    if not os.path.exists(dir):

        diagnal_is0_x = np.ones((self.No, self.No)) - np.eye(self.No)  #(200,200) 全1矩阵-单位阵
        diagnal_is0_y = np.ones((self.HNo, self.HNo)) - np.eye(self.HNo)  #(150,150) 全1矩阵-单位阵

        x = np.load(r'./dataself/elasticsearch/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)  # x是数据图 (100,200,200)
        y = np.load(r'./dataself/elasticsearch/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)  # y是label图 (100,150,150)
        IndexPaths_path = open(r'./dataself/elasticsearch/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')
        HunkIDmaps_path = open(r'./dataself/elasticsearch/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')
        IndexPaths = joblib.load(IndexPaths_path)
        HunkIDmaps = joblib.load(HunkIDmaps_path)

        node_x = np.zeros((x.shape[0], x.shape[1]))  # (100,200) 的零矩阵，100是batch大小

        bar1 = trange(x.shape[0])
        for i in bar1:  # 遍历100个图
            time.sleep(0.01)
            bar1.set_description('node_x[%i]' % i)  # 进度条左边显示信息
            node_x[i] = sum(x[i] * np.eye(self.No))  # x中第i个图 点乘取对角线矩阵，然后按行求和得到图x的各个节点的属性值
        node_x = node_x.reshape(x.shape[0], 1, self.No)  # (100,1,200)

        x_data = np.zeros((x.shape[0], x.shape[1], x.shape[1]))  # (100,200,200)
        y_label = np.zeros((y.shape[0], y.shape[1], y.shape[1]))  # (100,150,150)

        bar2 = trange(x.shape[0])
        for i in bar2:  # 遍历batch中每个图，i是batch中第i个图
            time.sleep(0.01)
            bar2.set_description('x_data[i,:,:] and y_label[i,:,:]')
            # x是数据，y是label
            x_data[i, :, :] = x[i, :, :] * diagnal_is0_x  # diagnal is removed，[:,:]是图的矩阵表示，点乘单位阵a
            y_label[i, :, :] = y[i, :, :] * diagnal_is0_y  # 去掉对角线的节点的属性值，只剩下边矩阵。i固定了，可以看到矩阵的原貌

        # entity 图 的源端、目的端、每条边
        Rr_data = np.zeros((100, self.No, self.Nr), dtype=float32);  # (100,200,310000) 每个entity节点对的 源端entity
        Rs_data = np.zeros((100, self.No, self.Nr), dtype=float32);  # (100,200,310000) 每个entity节点对的 目的端entity
        Ra_data = np.zeros((100, self.Dr, self.Nr), dtype=float32);  # (100,2,310000) batch_num, The Relationship Dimension, The Number of Relations

        # Hunk 图
        Hr_label = np.zeros((100, self.HNo, self.HNr), dtype=float32);  # (100,150,5402) 每个hunk节点对的 源端hunk
        Hs_label = np.zeros((100, self.HNo, self.HNr), dtype=float32);  # (100,150,5402) 每个hunk节点对的 目的端hunk
        Ra_label = np.zeros((100, self.Dr, self.HNr), dtype=float32);  # (100,2,5402) batch_num, The Relationship Dimension, The Number of Relations

        # 每条entity边 的 源端/目的端 hunk
        RHr_data = np.zeros((100, self.HNo, self.Nr), dtype=float32);  # (100,150,310000) 每个entity节点对的 源端hunk
        RHs_data = np.zeros((100, self.HNo, self.Nr), dtype=float32);  # (100,150,310000) 每个entity节点对的 目的端hunk

        # entity 图 的源端、目的端、每条边
        cnt = 0  # 遍历 number of relations
        # Relation Matrics R=<Rr,Rs,Ra>
        # 为每个relation 分配 源端i 和 目的端j
        bar3 = trange(self.No)
        for i in bar3:
            time.sleep(0.01)
            bar3.set_description('i in entity')
            for j in range(self.No):  # j 从 i+1 开始计算
                if (i != j):  # 每个图k的源端点和目的端点 Rr_data 和 Rs_data 都一样
                    # 每个entity节点对的 源端/目的端 entity
                    Rr_data[:, i, cnt] = 1.0;  # 200是源端点i (100,200,310000)  batch_num, The Number of Objects, The Number of Relations
                    Rs_data[:, j, cnt] = 1.0;  # 200是目的端点j (100,200,310000)  batch_num, The Number of Objects, The Number of Relations
                    for k in range(x_data.shape[0]):  # 遍历所有数据
                        # int(data[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                        # Ra_data (取自x) != Ra_label (取自y)。
                        Ra_data[k, int(x_data[k, i, j]), cnt] = 1  # (100,2,310000) batch_num, The Relationship Dimension, The Number of Relations
                    cnt += 1;

            # hunk 图 的源端、目的端、每条边
            cnt1 = 0  # 遍历 number of relations
            # Relation Matrics R=<Rr,Rs,Ra>
            # 为每个relation 分配 源端i 和 目的端j
            bar6 = trange(self.HNo)
            for i in bar6:
                time.sleep(0.01)
                bar6.set_description('i in hunk')
                for j in range(self.HNo):  # j 从 i+1 开始计算
                    if (i != j):  # 每个图k的源端点和目的端点 Rr_data 和 Rs_data 都一样
                        # 每个hunk节点对的 源端/目的端 hunk
                        Hr_label[:, i, cnt1] = 1.0;  # 150是源端点i (100,150,5402)  batch_num, The Number of Objects, The Number of Relations
                        Hs_label[:, j, cnt1] = 1.0;  # 150是目的端点j (100,150,5402)  batch_num, The Number of Objects, The Number of Relations
                        for k in range(y_label.shape[0]):  # 遍历batch
                            # int(y_label[k,i,j]) 是 0 或 1。3个维度是确定的，得到 Relationship Dimension
                            # Ra_data (取自x) != Ra_label (取自y)。
                            Ra_label[k, int(y_label[k, i, j]), cnt1] = 1  # (100,2,5402) batch_num, The Relationship Dimension, The Number of Relations
                        cnt1 += 1

        # cutting 为每个relation 匹配 源端hunk 和 目的端hunk
        bar8 = trange(x.shape[0])
        for index1 in bar8:  # 遍历100个图，每个图都有一个map
           time.sleep(0.01)
           bar8.set_description('IndexPath and HunkIDmap')
           # 识别i和j分别在哪个hunk里
           IndexPath = IndexPaths[index1]
           HunkIDmap = HunkIDmaps[index1]  # 是一个dict

           # 读取index
           index_txt = open(IndexPath)
           indexLines = index_txt.readlines()[:self.No]  # 按行读取，限制entity 标号大小
           # 分配源端、目的端 hunkIDnum
           cnt2 = 0
           for i in range(len(indexLines)):  # 有的entity图可能比(200,200)小，也有可能大
               for j in range(len(indexLines)):
                   if (i != j):
                       Hr_key = indexLines[i].strip()
                       Hs_key = indexLines[j].strip()
                       if (Hr_key != 'null'):
                           Hr_num = HunkIDmap[Hr_key]
                           if (Hr_num < self.HNo):  # 限制hunkID 标号大小
                               RHr_data[index1, Hr_num, cnt2] = 1.0
                       if (Hs_key != 'null'):
                           Hs_num = HunkIDmap[Hs_key]
                           if (Hs_num < self.HNo):
                               RHs_data[index1, Hs_num, cnt2] = 1.0
                       cnt2 += 1;

        # x中每个图的节点特征, 当总样本数是奇数的时候不可以除以2
        node_data_train = node_x[0:int(x.shape[0] / 2)]  # (50,1,200)
        node_data_test = node_x[int(x.shape[0] / 2):x.shape[0]]  # (50,1,200)

        # x中每个图的边的关系
        Ra_data_train = Ra_data[0:int(x.shape[0] / 2)]  # (50,2,310000)
        Ra_data_test = Ra_data[int(x.shape[0] / 2):x.shape[0]]  # (50,2,310000)

        # y中每个图的边的关系
        Ra_label_train = Ra_label[0:int(x.shape[0] / 2)]  # (50,2,5402)
        Ra_label_test = Ra_label[int(x.shape[0] / 2):x.shape[0]]  # (50,2,5402)


        # save RHr_data、RHs_data
        with open(r'./Intermediate_products/elasticsearch/node_data_train_' + str(step) + '.pkl', 'wb') as f21:
            joblib.dump(node_data_train, f21)

        with open(r'./Intermediate_products/elasticsearch/node_data_test_' + str(step) + '.pkl', 'wb') as f22:
            joblib.dump(node_data_test, f22)

        with open(r'./Intermediate_products/elasticsearch/Ra_data_train_' + str(step) + '.pkl', 'wb') as f23:
            joblib.dump(Ra_data_train, f23)

        with open(r'./Intermediate_products/elasticsearch/Ra_data_test_' + str(step) + '.pkl', 'wb') as f24:
            joblib.dump(Ra_data_test, f24)

        with open(r'./Intermediate_products/elasticsearch/Ra_label_train_' + str(step) + '.pkl', 'wb') as f25:
            joblib.dump(Ra_label_train, f25)

        with open(r'./Intermediate_products/elasticsearch/Ra_label_test_' + str(step) + '.pkl', 'wb') as f26:
            joblib.dump(Ra_label_test, f26)

        with open(r'./Intermediate_products/elasticsearch/Rr_data_' + str(step) + '.pkl', 'wb') as f27:
            joblib.dump(Rr_data, f27)
            # pkl.dump(RHr_data, f27)

        with open(r'./Intermediate_products/elasticsearch/Rs_data_' + str(step) + '.pkl', 'wb') as f28:
            joblib.dump(Rs_data, f28)
            # pkl.dump(RHs_data, f28)

        with open(r'./Intermediate_products/elasticsearch/Hr_label_' + str(step) + '.pkl', 'wb') as f29:
            joblib.dump(Hr_label, f29)

        with open(r'./Intermediate_products/elasticsearch/Hs_label_' + str(step) + '.pkl', 'wb') as f30:
            joblib.dump(Hs_label, f30)

        with open(r'./Intermediate_products/elasticsearch/RHr_data_' + str(step) + '.pkl', 'wb') as f31:
           joblib.dump(RHr_data, f31)
           # pkl.dump(RHr_data, f31)

        with open(r'./Intermediate_products/elasticsearch/RHs_data_' + str(step) + '.pkl', 'wb') as f32:
           joblib.dump(RHs_data, f32)
           # pkl.dump(RHs_data, f32)

    else:
        # RHr_data = pkl.load(open(r'./Intermediate_products/elasticsearch/RHr_data_' + str(step) + '.pkl', 'rb'))
        # RHs_data = pkl.load(open(r'./Intermediate_products/elasticsearch/RHs_data_' + str(step) + '.pkl', 'rb'))
        node_data_train = joblib.load(open(r'./Intermediate_products/elasticsearch/node_data_train_' + str(step) + '.pkl', 'rb'))
        node_data_test = joblib.load(open(r'./Intermediate_products/elasticsearch/node_data_test_' + str(step) + '.pkl', 'rb'))
        Ra_data_train = joblib.load(open(r'./Intermediate_products/elasticsearch/Ra_data_train_' + str(step) + '.pkl', 'rb'))
        Ra_data_test = joblib.load(open(r'./Intermediate_products/elasticsearch/Ra_data_test_' + str(step) + '.pkl', 'rb'))
        Ra_label_train = joblib.load(open(r'./Intermediate_products/elasticsearch/Ra_label_train_' + str(step) + '.pkl', 'rb'))
        Ra_label_test = joblib.load(open(r'./Intermediate_products/elasticsearch/Ra_label_test_' + str(step) + '.pkl', 'rb'))
        Rr_data = joblib.load(open(r'./Intermediate_products/elasticsearch/Rr_data_' + str(step) + '.pkl', 'rb'))
        Rs_data = joblib.load(open(r'./Intermediate_products/elasticsearch/Rs_data_' + str(step) + '.pkl', 'rb'))
        Hr_label = joblib.load(open(r'./Intermediate_products/elasticsearch/Hr_label_' + str(step) + '.pkl', 'rb'))
        Hs_label = joblib.load(open(r'./Intermediate_products/elasticsearch/Hs_label_' + str(step) + '.pkl', 'rb'))
        RHr_data = joblib.load(open(r'./Intermediate_products/elasticsearch/RHr_data_' + str(step) + '.pkl', 'rb'))
        RHs_data = joblib.load(open(r'./Intermediate_products/elasticsearch/RHs_data_' + str(step) + '.pkl', 'rb'))

    return node_data_train,node_data_test,\
           Ra_data_train,Ra_data_test,\
           Ra_label_train,Ra_label_test,\
           Rr_data,Rs_data,\
           Hr_label,Hs_label,\
           RHr_data,RHs_data

