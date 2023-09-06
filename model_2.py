# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

import numpy as np
import time
import os
from utils2 import read_data
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr


def process_edge(Ra):
    for i in range(len(Ra)):
        for j in range(len(Ra[i])):
            for k in range(len(Ra[i][j])):
                Ra[i][j][k] = Ra[i][j][k]
    return Ra


def process_node(O):
    for i in range(len(O)):
        for j in range(len(O[i])):
            for k in range(len(O[i][j])):
                O[i][j][k] = round(O[i][j][k])
    return O

# 按道理每个图都有一个准确率
def top_ACC(Ra, Ra_t):
    # Ra=O.reshape(O.shape[0],O.shape[2])
    # Ra_t=Ra_t.reshape(O_t.shape[0],O_t.shape[2])
    count = 0
    for i in range(Ra.shape[0]):
        for j in range(Ra.shape[2]):
            Ra_t[i, 1, j] = Ra_t[i, 1, j]
            Ra_t[i, 0, j] = Ra_t[i, 0, j]
            if np.argmax(Ra_t[i, :, j]) == np.argmax(Ra[i, :, j]):
                count += 1
    return float(count / (Ra.shape[0] * Ra.shape[2]))


def node_ACC(O, O_t):
    O = O.reshape(O.shape[0], O.shape[2])
    O_t = O_t.reshape(O_t.shape[0], O_t.shape[2])
    count = 0
    for i in range(len(O)):
        for j in range(len(O[i])):
            if O_t[i][j] > 5:
                O_t[i][j] = 1
            else:
                O_t[i][j] = 0
            if O[i][j] > 5:
                O[i][j] = 1
            else:
                O[i][j] = 0
            if O_t[i][j] == O[i][j]:
                count += 1
    return float(count / (O.shape[0] * O.shape[1]))

# 测量预测值Ŷ与某些真实值匹配程度。
# MSE 通常用作回归问题的损失函数。例如，根据其属性估算公寓的价格
def mse(label, real):
    mse = 0
    for i in range(label.shape[0]):
        score = mean_squared_error(label[i, 0, :], real[i, 0, :])
        mse += score
    return mse / label.shape[0]

# R2指的是相关系数，一般机器默认的是R2>0.99，这样才具有可行度和线性关系。
def r2(label, real):
    r2 = 0
    for i in range(label.shape[0]):
        score = r2_score(label[i, 0, :], real[i, 0, :])
        r2 += score
    return r2 / label.shape[0]

# 是用于度量两个变量X和Y之间的相关（线性相关），其值介于-1与1之间
# 计算出变量A和变量B的皮尔逊相关系数为0，不代表A和B之间没有相关性，只能说明A和B之间不存在线性相关关系。
def pear(label, real):
    p = 0
    for i in range(label.shape[0]):
        score = pearsonr(label[i, 0, :], real[i, 0, :])[0]
        p += score
    return p / label.shape[0]


def spear(label, real):
    sp = 0
    for i in range(label.shape[0]):
        score = spearmanr(label[i, 0, :], real[i, 0, :])[0]
        sp += score
    return sp / label.shape[0]


class graph2graph(object):
    def __init__(self, sess, Ds, No, HNo, Nr, HNr, Dr, De_o, De_r, Mini_batch, checkpoint_dir, epoch, Ds_inter,
                 Dr_inter, Step):
        self.sess = sess
        self.Ds = Ds
        self.No = No
        self.HNo = HNo
        self.Nr = Nr
        self.HNr = HNr
        self.Dr = Dr
        self.Ds_inter = Ds_inter
        self.Dr_inter = Dr_inter
        # self.Dx=Dx
        self.De_o = De_o
        self.De_r = De_r
        self.mini_batch_num = Mini_batch
        self.epoch = epoch
        # batch normalization : deals with poor initialization helps gradient flow
        self.checkpoint_dir = checkpoint_dir
        self.Step = Step
        self.build_model()

    def variable_summaries(self, var, idx):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries_' + str(idx)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_model(self):  # output 是 O_2 和 Ra_2

        # tf.placeholder 先占坑，在sess.run 运行结果的时候，再给他具体的值
        # tf.placeholder 和 feed_dict绑定
        self.O_1 = tf.placeholder(tf.float32, [self.mini_batch_num, self.Ds, self.No],
                                  name="node_data_train")  # node_data_train, (50,1,200) data图 batch_num, The State Dimention, The Number of Objects
        self.O_target = tf.placeholder(tf.float32, [self.mini_batch_num, self.Ds, self.No],
                                       name="node_label_train")  # node_label_train, (50,1,200) label图

        # Relation Matrics R=<Rr,Rs,Ra>
        # entity graph 的 源端、目的端、边
        self.Rr = tf.placeholder(tf.float32, [self.mini_batch_num, self.No, self.Nr],
                                 name="Rr_data")  # Rr_data, (50,200,39800) 每个节点对的源端 batch_num, The Number of Objects, The Number of Relations
        self.Rs = tf.placeholder(tf.float32, [self.mini_batch_num, self.No, self.Nr],
                                 name="Rs_data")  # Rs_data, (50,200,39800) 每个节点对的目的端
        self.Ra_1 = tf.placeholder(tf.float32, [self.mini_batch_num, self.Dr, self.Nr],
                                   name="Ra_data_train")  # Ra_data_train, (50,2,39800) data图每个节点对是否有边 batch_num, The Relationship Dimension, The Number of Relations

        # hunk graph 的 源端、目的端、边
        self.Hr = tf.placeholder(tf.float32, [self.mini_batch_num, self.HNo, self.HNr],
                                 name="Hr_label")  # Hr_label, (50,74,5402) 每个节点对的源端 batch_num, The Number of Objects, The Number of Relations
        self.Hs = tf.placeholder(tf.float32, [self.mini_batch_num, self.HNo, self.HNr],
                                 name="Hs_label")  # Hs_label, (50,74,5402) 每个节点对的目的端
        self.Ra_target = tf.placeholder(tf.float32, [self.mini_batch_num, self.Dr, self.HNr],
                                        name="Ra_label_train")  # Ra_label_train, (50,2,5402) label图每个节点对是否有边
        # External Effects
        # self.X = tf.placeholder(tf.float32, [self.mini_batch_num,self.Dx,self.No], name="X")

        # entity 边源端、目的端的hunk
        self.RHr = tf.placeholder(tf.float32, [self.mini_batch_num, self.HNo, self.Nr],
                                  name="RHr_data")  # RHr_data, (50,74,39800)
        self.RHs = tf.placeholder(tf.float32, [self.mini_batch_num, self.HNo, self.Nr],
                                  name="RHs_data")  # RHs_data, (50,74,39800)

        # step1:
        # marshalling function !!!! 主要改这个B_1、self.m
        self.B_1 = self.m_B1(self.O_1, self.Rr, self.Rs, self.Ra_1)  # (50,4,39800) 其中4=1(源端节点)+1(目的端节点)+2(边)

        # updating the entity state (node translation)
        self.E_O_1 = self.phi_E_O_1(self.B_1) # mlp函数 (50,20,39800)
        self.C_O_1 = self.a_O(self.E_O_1,self.Rr,self.O_1) # aggregation函数 (50,21,200)
        self.O_2 = self.phi_U_O_1(self.C_O_1)  # 降维, data图 (50,1,200)

        # marshalling function
        self.B_2 = self.m_B1(self.O_2, self.Rr, self.Rs, self.Ra_1)  # 和 self.B_1 计算过程一样
        self.B_3 = self.m_B2(self.B_2, self.RHr, self.RHs, self.Hr, self.Hs, self.Ra_target)  # !!!
        #
        # #updating the edge (edge translation) !!!!!!主要改 phi_E_R_1
        # self.E_R_1 = self.phi_E_R_1(self.B_1)  # (50,20,39800) add a constrain to make edge1-2 == edge 2-1
        # self.C_R_1 = self.a_R(self.E_R_1,self.Ra_1) # aggregation函数 (50,20,39800) + (50,2,39800)
        # self.Ra_2, self.Ra_logits_2 = self.phi_U_R_1(self.C_R_1) # 降维, score被softmax后(50,2,39800), 输出的score(50,2,39800)

        # updating the hunk edge (edge translation) self.phi_B2 和 self.phi_HRa 最重要
        self.H_Ra = self.phi_B2(self.B_3)  # (50,20,5402)
        self.H_Ra1 = self.a_R(self.H_Ra, self.Ra_target)  # (50,22,5402) aggregation函数 (50,20,5402) 和 (50,2,5402) 在第2个维度拼接
        self.H_Ra2, self.H_Ra2_logits = self.phi_HRa(self.H_Ra1)  # 降维, score被softmax后(50,2,39800), 输出的score(50,2,39800)

        # loss
        # 计算平方差
        # # tf.reduce_max, axis=0 指的是计算矩阵每列的最大值，axis=1 计算行最大值
        # # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
        # self.loss_node_mse = tf.reduce_mean(tf.reduce_mean(tf.square(self.O_2-self.O_target)/(tf.reduce_max(self.O_target)),[1,2])) # 比较O_2(node_data_train) 和 O_target(node_label_train)
        # # 调公式计算loss, logits 是神经网络最后一层的输出, labels 是实际的标签，大小同上
        # self.loss_edge_mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.Ra_logits_2,labels=self.Ra_target,dim=1)) # 比较Ra_logits_2(Ra_data_train) 和 Ra_target(Ra_label_train)

        self.loss_Hedge_mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.H_Ra2_logits,
            labels=self.Ra_target,
            dim=1))  # 比较Ra_logits_2(Ra_data_train) 和 Ra_target(Ra_label_train)

        # 在train 中 要额外加入的 loss
        self.loss_map, self.theta = self.map_loss(2)
        self.loss_E_HR = 0.001 * tf.nn.l2_loss(self.H_Ra)
        params_list = tf.global_variables()  # 全部变量

        # tf 相关
        for i in range(len(params_list)):
            self.variable_summaries(params_list[i], i)
        self.loss_para = 0
        for i in params_list:
            self.loss_para += 0.001 * tf.nn.l2_loss(i);

        # 这个方法是添加变量到直方图中，启动 tensorborder 可以看
        tf.summary.scalar('hunk_edge_mse', self.loss_Hedge_mse)
        tf.summary.scalar('map_mse', self.loss_map)

        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars]

        self.saver = tf.train.Saver()

    def m_B1(self, O_1, Rr, Rs, Ra_1):
        # tf.concat代表在第1个维度(列)拼接
        # tf.matmul 将矩阵a 乘以矩阵b,生成a*b
        return tf.concat([tf.matmul(O_1, Rr), tf.matmul(O_1, Rs), Ra_1], 1);

    def m_B2(self, B_1, RHr, RHs, Hr, Hs, Ra_target):
        B_t = tf.transpose(B_1, [0, 2, 1])  # (50,4,39800)->(50,39800,4)
        Hr_neighbors_e = tf.matmul(RHr, B_t)  # (50,74,39800)*(50,39800,4)=(50,74,4)
        Hs_neighbors_e = tf.matmul(RHs, B_t)  # (50,74,39800)*(50,39800,4)=(50,74,4)
        neighbors = Hr_neighbors_e + Hs_neighbors_e  # (50,74,4)

        Hr_t = tf.transpose(Hr, [0, 2, 1])  # (50,74,5402)->(50,5402,74)
        Hs_t = tf.transpose(Hs, [0, 2, 1])  # (50,74,5402)->(50,5402,74)
        Hr_neighbors_h = tf.matmul(Hr_t, neighbors)  # (50,5402,74)*(50,74,4)=(50,5402,4)
        Hs_neighbors_h = tf.matmul(Hs_t, neighbors)  # (50,5402,74)*(50,74,4)=(50,5402,4)
        Hr_neighbors_ht = tf.transpose(Hr_neighbors_h, [0, 2, 1])  # (50,5402,4)->(50,4,5402)
        Hs_neighbors_ht = tf.transpose(Hs_neighbors_h, [0, 2, 1])  # (50,5402,4)->(50,4,5402)
        B_2 = tf.concat([Hr_neighbors_ht, Hs_neighbors_ht, Ra_target],1);  # concat (50,4,5402),(50,4,5402),(50,2,5402)的第1个维度=(50,10,5402)
        return B_2;

    def phi_E_O_1(self, B):
        with tf.variable_scope("phi_E_O1") as scope:
            h_size = 20;  # (50,4,39800) —> (50,20,39800)
            B_trans = tf.transpose(B, [0, 2, 1]);  # (50,39800,4)
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.Nr, (2 * self.Ds + self.Dr)]);
            # 要定义成变量，它才是一个变量
            w1 = tf.Variable(tf.truncated_normal([(2 * self.Ds + self.Dr), h_size], stddev=0.1), name="r1_w1o",
                             dtype=tf.float32);
            b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1o", dtype=tf.float32);
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);

            w5 = tf.Variable(tf.truncated_normal([h_size, self.De_o], stddev=0.1), name="r1_w5o", dtype=tf.float32);
            b5 = tf.Variable(tf.zeros([self.De_o]), name="r1_b5o", dtype=tf.float32);
            # h5 = tf.nn.relu(tf.matmul(h1, w5) + b5);
            h5 = tf.matmul(h1, w5) + b5;

            h5_trans = tf.reshape(h5, [self.mini_batch_num, self.Nr, self.De_o]);
            h5_trans = tf.transpose(h5_trans, [0, 2, 1]);  # (50,20,39800)
            return (h5_trans);

    def a_O(self, E, Rr, O):
        # E:(50,20,39800)
        # O: (50,1,200)
        # tf.transpose(Rr,[0,2,1]) #(50,39800,200)
        # tf.transpose(self.Rs,[0,2,1]) #(50,39800,200)
        E_bar = tf.matmul(E, tf.transpose(Rr, [0, 2, 1])) + tf.matmul(E, tf.transpose(self.Rs, [0, 2,
                                                                                                1]));  # (50,20,200) 把39800抵消了，也即根据边找到对应端点的新的值
        return (tf.concat([O, E_bar], 1));  # (50,20,200) + (50,1,200)

    def phi_U_O_1(self, C):
        with tf.variable_scope("phi_U_O1") as scope:
            h_size = 20;
            C_trans = tf.transpose(C, [0, 2, 1]); #
            C_trans = tf.reshape(C_trans, [self.mini_batch_num * self.No, (self.Ds + self.De_o)]);
            # 要定义成变量，它才是一个变量
            w1 = tf.Variable(tf.truncated_normal([(self.Ds + self.De_o), h_size], stddev=0.1), name="o1_w1o",
                             dtype=tf.float32);
            b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1o", dtype=tf.float32);
            h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
            w2 = tf.Variable(tf.truncated_normal([h_size, self.Ds], stddev=0.1), name="o1_w2o", dtype=tf.float32);
            b2 = tf.Variable(tf.zeros([self.Ds_inter]), name="o1_b2o", dtype=tf.float32);
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.No, self.Ds_inter]);
            h2_trans = tf.transpose(h2_trans, [0, 2, 1]);
            return (h2_trans);

    def phi_E_R_1(self, B):
        with tf.variable_scope("phi_E_R1") as scope:
            h_size = 20;  # B: (50,4,39800)
            B_trans = tf.transpose(B, [0, 2, 1]);  # (50,39800,4)
            # Nr:39800 Number of Relations; Ds:1 State Dimention; Dr:2 Relationship Dimension
            # (39800*50, 1+1+2) = (78000,4)
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.Nr, (2 * self.Ds + self.Dr)]);

            # 要定义成变量，它才是一个变量;
            # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None):
            # shape 表示生成张量的维度（a * a），mean是均值，stddev 是标准差

            # w1_1:(1,20) 每个节点的权重; w1_2:(2,20) 每个边的权重
            w1_1 = tf.Variable(tf.truncated_normal([(self.Ds), h_size], stddev=0.1), name="r1_w1r1", dtype=tf.float32);
            w1_2 = tf.Variable(tf.truncated_normal([(self.Dr), h_size], stddev=0.1), name="r1_w1r2", dtype=tf.float32);
            w1 = tf.concat([w1_1, w1_1, w1_2], 0)  # (4,20)
            # w1 = tf.Variable(tf.truncated_normal([(2*self.Ds+self.Dr), h_size], stddev=0.1), name="r_w1r", dtype=tf.float32);
            b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1r", dtype=tf.float32);  # (20)
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);  # (78000,4)*(4,20) = (78000,20) 每个batch中所有边的维度

            # w2:(20,20); De_r:20 Effect Dimension on edge;
            w2 = tf.Variable(tf.truncated_normal([h_size, self.De_r], stddev=0.1), name="r1_w2r", dtype=tf.float32);
            b2 = tf.Variable(tf.zeros([self.De_r]), name="r1_b2r", dtype=tf.float32);  # (20)
            h2 = tf.matmul(h1, w2) + b2;  # (78000,20) * (20,20) = (78000,20)
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.Nr, self.De_r]);  # (50,39800,20)
            h2_trans = tf.transpose(h2_trans, [0, 2, 1]);  # (50,20,39800) 每个B的特征

            # edge translation
            # tf.transpose(self.Rr,[0,2,1]):(50,39800,200)
            # tf.transpose(self.Rs,[0,2,1]): (50,39800,200)
            h2_trans_bar1 = tf.matmul(h2_trans, tf.transpose(self.Rr, [0, 2, 1]));  # (50,20,200) B根据源端点排序 (50,20,39800)*(50,39800,200)
            h2_trans_bar2 = tf.matmul(h2_trans, tf.transpose(self.Rs, [0, 2, 1]));  # (50,20,200) B根据目的端点排序 (50,20,39800)*(50,39800,200)
            # (50,20,200)*(50,200,39800) = (50,20,39800) 源端点和目的端点分别还原成关系的矩阵，结合源端点和目的端点的B
            effects = tf.matmul(h2_trans_bar1, self.Rr) + tf.matmul(h2_trans_bar2, self.Rs)

            return effects

    # 对hunk边进行编码的mlp
    def phi_B2(self, B2):
        with tf.variable_scope("phi_B2") as scope:
            h_size = 20;  # B2: (50,10,5402)
            B_trans = tf.transpose(B2, [0, 2, 1]);  # (50,5402,10)
            # HNr:5402 Number of Relations; Ds:1 State Dimention; Dr:2 Relationship Dimension
            # (5402*50, 10) = (273800,10)
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.HNr, B2.shape[1]]);

            # 要定义成变量，它才是一个变量;
            # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None):
            # shape 表示生成张量的维度（a * a），mean是均值，stddev 是标准差

            w1 = tf.Variable(tf.truncated_normal([B2.shape[1], h_size], stddev=0.1), name="w1",
                             dtype=tf.float32);  # (10,20)
            b1 = tf.Variable(tf.zeros([h_size]), name="b1", dtype=tf.float32);  # (20)
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);  # (273800,10)*(10,20) = (273800,20) 每个batch中所有hunk边的维度

            # w2:(20,20); De_r:20 Effect Dimension on edge;
            w2 = tf.Variable(tf.truncated_normal([h_size, self.De_r], stddev=0.1), name="r1_w2r", dtype=tf.float32);
            b2 = tf.Variable(tf.zeros([self.De_r]), name="b2", dtype=tf.float32);  # (20)
            h2 = tf.matmul(h1, w2) + b2;  # (273800,20) * (20,20) = (273800,20)
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.HNr, self.De_r]);  # (50,5402,20)
            h2_trans = tf.transpose(h2_trans, [0, 2, 1]);  # (50,20,5402) 每个B的特征

            # edge translation
            # tf.transpose(self.Rr,[0,2,1]):(50,5402,20)
            # tf.transpose(self.Rs,[0,2,1]):(50,5402,20)
            h2_trans_bar1 = tf.matmul(h2_trans, tf.transpose(self.Hr, [0, 2, 1]));  # (50,20,74) B2根据源端hunk排序 (50,20,5402)*(50,5402,74)
            h2_trans_bar2 = tf.matmul(h2_trans, tf.transpose(self.Hs, [0, 2, 1]));  # (50,20,74) B2根据目的端hunk排序 (50,20,5402)*(50,5402,74)
            # (50,20,74)*(50,74,5402) = (50,20,5402) 源端hunk 和 目的hunk 分别还原成关系的矩阵，结合源端hunk 和 目的端hunk的 B2
            effects = tf.matmul(h2_trans_bar1, self.Hr) + tf.matmul(h2_trans_bar2, self.Hs)

            return effects  # (50,20,5402)

    def a_R(self, E, Ra):
        C_R = tf.concat([Ra, E], 1)
        return (C_R);

    def phi_U_R_1(self, C_R):
        with tf.variable_scope("phi_U_R1") as scope:
            h_size = 20;
            C_trans = tf.transpose(C_R, [0, 2, 1]);
            C_trans = tf.reshape(C_trans, [self.mini_batch_num * self.Nr, (self.De_r + self.Dr)]);

            # 要定义成变量，它才是一个变量
            w1 = tf.Variable(tf.truncated_normal([(self.De_r + self.Dr), h_size], stddev=0.1), name="o1_w1r",
                             dtype=tf.float32);
            b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1r", dtype=tf.float32);
            h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);

            w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr_inter], stddev=0.1), name="o1_w2r", dtype=tf.float32);
            b2 = tf.Variable(tf.zeros([self.Dr_inter]), name="o1_b2r", dtype=tf.float32);

            h2_trans = tf.reshape(tf.matmul(h1, w2) + b2, [self.mini_batch_num, self.Nr, self.Dr]);  # (50,39800,2)
            h2_trans_logits = tf.transpose(h2_trans, [0, 2, 1]);  # (50,2,39800)
            h2 = tf.nn.softmax(h2_trans_logits, dim=1)  # (50,2,39800)
            return h2, h2_trans_logits  # score被softmax后，输出的score

    # 给mlp的输出的hunk边的表示 降维
    def phi_HRa(self, HRa):
        with tf.variable_scope("phi_U_R1") as scope:
            h_size = 20;
            HRa_trans = tf.transpose(HRa, [0, 2, 1]);  # (50,22,5402)->(50,5402,22)
            HRa_trans2 = tf.reshape(HRa_trans, [self.mini_batch_num * self.HNr, HRa.shape[1]]);  # (50*5402,22)

            # 要定义成变量，它才是一个变量
            w1 = tf.Variable(tf.truncated_normal([HRa.shape[1], h_size], stddev=0.1), name="HRa_w1",
                             dtype=tf.float32);  # (22,20)
            b1 = tf.Variable(tf.zeros([h_size]), name="HRa_b1", dtype=tf.float32);  # (20)
            h1 = tf.nn.relu(
                tf.matmul(HRa_trans2, w1) + b1);  # (273800,22)*(22,20) + 20 = (273800,20) 每个batch中所有hunk边的维度

            w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr_inter], stddev=0.1), name="o1_w2r",
                             dtype=tf.float32);  # (20,2)
            b2 = tf.Variable(tf.zeros([self.Dr_inter]), name="o1_b2r", dtype=tf.float32);  # (2)

            h2_trans = tf.reshape(tf.matmul(h1, w2) + b2, [self.mini_batch_num, self.HNr, self.Dr]);  # (50,5402,2)
            h2_trans_logits = tf.transpose(h2_trans, [0, 2, 1]);  # (50,2,5402)
            h2 = tf.nn.softmax(h2_trans_logits, dim=1)  # (50,2,5402)
            return h2, h2_trans_logits  # score被softmax后的结果，输出的score

    def chebyshev_polynomials(self, Ra, k):
        """Calculate Chebyshev polynomials up to order k."""
        Ra = tf.argmax(Ra, axis=1)
        Ra = tf.cast(Ra, tf.float32)
        Ra = tf.reshape(Ra, [self.mini_batch_num, 1, self.Nr])
        # transofrm to adjacent matrix
        s0 = np.zeros((Ra.shape[0], 1, self.Nr)).astype(np.float32)
        s0[:, :, 0:self.No - 1] = 1
        S = tf.multiply(s0, Ra)

        for i in range(1, self.No):
            s = np.zeros((Ra.shape[0], 1, self.Nr)).astype(np.float32)
            s[:, :, i * (self.No - 1):i * (self.No - 1) + (self.No - 1)] = 1
            S = tf.concat([S, tf.multiply(s, Ra)], 1)  # S is the arrange adjacent vector with related node unmasked
        T = np.zeros((Ra.shape[0], self.Nr, self.No)).astype(np.float32)
        for i in range(self.No):
            t = np.zeros((Ra.shape[0], self.No - 1, self.No))
            for j in range(0, i):
                t[:, j, j] = 1
            for j in range(i, self.No - 1):
                t[:, j, j + 1] = 1
            T[:, i * (self.No - 1):i * (self.No - 1) + self.No - 1, :] = t
        adj = tf.matmul(S, T)

        def normalize_adj(adj):
            """Symmetrically normalize adjacency matrix."""
            rowsum = tf.reduce_sum(adj, 2) + 0.001 * np.ones((adj.shape[0], adj.shape[1])).astype(np.float32)
            power = (-0.5) * np.ones((rowsum.shape[0], rowsum.shape[1])).astype(np.float32)
            d_inv_sqrt = tf.pow(rowsum, power)
            d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
            a = tf.transpose(tf.matmul(adj, d_mat_inv_sqrt), [0, 2, 1])
            return tf.matmul(a, d_mat_inv_sqrt)

        adj_normalized = normalize_adj(adj)
        I = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2])).astype(np.float32)
        for i in range(adj.shape[0]):
            I[i] = np.eye(adj.shape[1])
        laplacian = I - adj_normalized

        largest_eigval = 1.5 * np.ones((adj.shape[0], 1)).astype(
            np.float32)  # tf.reduce_max(tf.self_adjoint_eig(laplacian)[0],1)
        eig_ = tf.divide(2 * np.ones(largest_eigval.shape).astype(np.float32), largest_eigval)
        eig_ = tf.reshape(eig_, [I.shape[0], 1, 1])
        scaled_laplacian = tf.multiply(tf.tile(eig_, multiples=[1, I.shape[1], I.shape[2]]), laplacian) - I

        t_k = tf.concat([tf.reshape(I, [I.shape[0], 1, I.shape[1], I.shape[2]]),
                         tf.reshape(scaled_laplacian, [I.shape[0], 1, I.shape[1], I.shape[2]])], axis=1)

        def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, s_lap):
            s_lap_new = 2 * tf.matmul(s_lap, t_k_minus_one) - t_k_minus_two
            return s_lap_new

        for i in range(2, k):
            t_k_ = chebyshev_recurrence(t_k[:, -1, :, :], t_k[:, -2, :, :], scaled_laplacian)
            t_k = tf.concat([t_k, tf.reshape(t_k_, [I.shape[0], 1, I.shape[1], I.shape[2]])], axis=1)

        return t_k

    def map_conv(self, theta, Ra, O, k):
        t_k = self.chebyshev_polynomials(Ra, k)
        theta_norm = tf.nn.softmax(theta, dim=1)
        theta_norm1 = tf.tile(theta_norm, multiples=[self.mini_batch_num, 1, self.No, self.No])
        L = tf.reduce_sum(tf.multiply(theta_norm1, t_k), 1)
        O_trans = tf.transpose(O, [0, 2, 1])
        conv = tf.matmul(O, L)
        loss = tf.matmul(conv, O_trans)
        return tf.reduce_mean(tf.square(loss) / self.Nr)

    def map_loss(self, k):
        with tf.variable_scope("map_conv") as scope:
            # 要定义成变量，它才是一个变量
            theta1 = tf.Variable(tf.truncated_normal([1, k, 1, 1], stddev=0.1), name="map_theta1", dtype=tf.float32);
            theta2 = tf.Variable(tf.truncated_normal([1, k, 1, 1], stddev=0.1), name="map_theta2", dtype=tf.float32);
            loss2 = tf.sqrt(2 * tf.nn.l2_loss(tf.reshape(theta2, [k]))) + tf.sqrt(
                tf.nn.l2_loss(tf.reshape(theta1, [k])) * 2)
            return 0.01 * loss2, theta2

    def train(self, args):  # 激活(sess.run) model里面的参数
        train_loss = 10 * self.loss_Hedge_mse + 0.1 * self.loss_map + self.loss_para
        optimizer = tf.train.AdamOptimizer(0.0003);  # 优化器的主要作用就是根据损失函数求出的loss，对神经网络的参数进行更新
        trainer = optimizer.minimize(train_loss);  # 优化器(损失函数)

        init_op = tf.global_variables_initializer()  # 初始化所有变量, 此时还没激活
        self.sess.run(init_op)  # sess就像一个指针，处理的地方被激活

        # read data
        node_data_train, node_data_test, \
        Ra_data_train, Ra_data_test, \
        Ra_label_train, Ra_label_test, \
        Rr_data, Rs_data, \
        Hr_label, Hs_label, \
        RHr_data, RHs_data = read_data(self, self.Step)

        max_epoches = self.epoch
        counter = 1

        # tensorboard
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("D:/logs/")

        with open(r'outputSelf/elasticsearch/result_' + str(self.Step) + '.npy', "a", encoding='utf-8') as f:
            for i in range(max_epoches):
                tr_loss_Hedge = 0
                tr_loss_map = 0
                HRa_t = []

                for j in range(int(len(node_data_train) / self.mini_batch_num)):  # 按照batch划分轮数
                    batch_O = node_data_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num];  # (50,1,200) 取一个batch
                    batch_Ra = Ra_data_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]  # (50,2,39800)
                    batch_Ra_target = Ra_label_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]  # (50,2,39800)sess就像一个指针，处理的地方被激活

                    Merge, HRa_t_batch, tr_loss_part_Hedge, tr_loss_part_map, theta, _ = \
                        self.sess.run(
                            [merged, self.H_Ra2, self.loss_Hedge_mse, self.loss_map, self.theta, trainer],
                            feed_dict={
                                self.O_1: batch_O,
                                self.Ra_1: batch_Ra,
                                self.Ra_target: batch_Ra_target,
                                self.Rr: Rr_data[:self.mini_batch_num],
                                self.Rs: Rs_data[:self.mini_batch_num],
                                self.Hr: Hr_label[:self.mini_batch_num],
                                self.Hs: Hs_label[:self.mini_batch_num],
                                self.RHr: RHr_data[:self.mini_batch_num],
                                self.RHs: RHs_data[:self.mini_batch_num]
                            }
                        );

                    tr_loss_Hedge += tr_loss_part_Hedge
                    tr_loss_map += tr_loss_part_map
                    HRa_t.append(HRa_t_batch)

                    # Tensorboard
                    writer.add_summary(Merge,j)

                acc_top = top_ACC(Ra_label_train, np.array(HRa_t).reshape(Ra_label_train.shape[0], Ra_label_train.shape[1],
                                                                          Ra_label_train.shape[2]))  # (50,2,5402)
                theta = theta.reshape([2])

                resultString = "Epoch " + str(i + 1) + \
                               " acc: " + str(acc_top)[0:6] + \
                               " Hedge loss: " + str(tr_loss_Hedge / (int(len(node_data_train) / self.mini_batch_num)))[0:6] + \
                               " map MSE: " + str(tr_loss_map / (int(len(node_data_train) / self.mini_batch_num)))[0:6] + \
                               " theta: " + str(theta[0]) + ' ' + str(theta[1]) + '\n'

                with open(r'outputSelf/elasticsearch/result_' + str(self.Step) + '.npy', "a", encoding='utf-8') as f:
                    f.write(resultString) # 存储实验结果

                print(resultString)

                counter += 1
                self.save(args.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):

        node_data_train, node_data_test, \
        Ra_data_train, Ra_data_test, \
        Ra_label_train, Ra_label_test, \
        Rr_data, Rs_data, \
        Hr_label, Hs_label, \
        RHr_data, RHs_data = read_data(self, self.Step)

        init_op = tf.global_variables_initializer()  # 初始化所有变量, 此时还没激活
        self.sess.run(init_op)  # sess就像一个指针，处理的地方被激活
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        te_loss_Hedge = 0
        te_loss_map = 0
        HRa_t = []
        '''
        node_data_test=node_data_train
        node_label_test=node_label_train
        Ra_data_test=Ra_data_train
        Ra_label_test=Ra_label_train
        '''
        for j in range(int(len(node_data_test) / self.mini_batch_num)):
            batch_O = node_data_test[j * self.mini_batch_num:(j + 1) * self.mini_batch_num];
            batch_Ra = Ra_data_test[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]
            batch_Ra_target = Ra_label_test[
                              j * self.mini_batch_num:(j + 1) * self.mini_batch_num]  # sess就像一个指针，处理的地方被激活

            # feed_dict 是字典形式{key:value，key:value,...}
            te_loss_part_Hedge, te_loss_part_map, HRa_part = \
                self.sess.run([self.loss_Hedge_mse,
                               self.loss_map,
                               self.H_Ra2
                               ],
                              feed_dict={
                                  self.O_1: batch_O,
                                  self.Ra_1: batch_Ra,
                                  self.Ra_target: batch_Ra_target,
                                  self.Rr: Rr_data[:self.mini_batch_num],
                                  self.Rs: Rs_data[:self.mini_batch_num],
                                  self.Hr: Hr_label[:self.mini_batch_num],
                                  self.Hs: Hs_label[:self.mini_batch_num],
                                  self.RHr: RHr_data[:self.mini_batch_num],
                                  self.RHs: RHs_data[:self.mini_batch_num]
                              }
                              );

            te_loss_Hedge += te_loss_part_Hedge
            te_loss_map += te_loss_part_map

            HRa_t.append(HRa_part)  # 来源于Ra_2
        HRa_t = np.array(HRa_t).reshape(len(node_data_test), self.Dr, self.HNr)  # (50,1,5402)

        np.save('outputSelf/elasticsearch/HRa_t' + str(self.No) + '.npy',
                np.array(HRa_t).reshape(len(node_data_test), self.Dr, self.HNr))
        np.save('outputSelf/elasticsearch/HRa_y' + str(self.No) + '.npy',
                Ra_label_test.reshape(len(node_data_test), self.Dr, self.HNr))

        # evaluate, 从这里可以看出，最终生成的只有 O_t 和 Ra_t, t 是 target 的意思
        HRa_t = process_edge(HRa_t)  # ?

        print('mse-edge: ' + str(mse(Ra_label_test, HRa_t)))
        print('r2-edge: ' + str(r2(Ra_label_test, HRa_t)))
        print('p-edge: ' + str(pear(Ra_label_test, HRa_t)))
        print('sp-edge: ' + str(spear(Ra_label_test, HRa_t)))
        print('topol_acc: ' + str(top_ACC(Ra_label_test, HRa_t)))

        # print('mse-node2: '+str(mse(node_label_test,node_data_test)))
        # print('topol_acc2: '+str(top_ACC(Ra_data_test,Ra_label_test)))

