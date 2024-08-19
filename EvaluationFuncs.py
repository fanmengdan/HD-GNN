from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def process_edge(Ra): # 如何遍历 edge 的结构, 备用函数
    for i in range(len(Ra)):
        for j in range(len(Ra[i])):
            for k in range(len(Ra[i][j])):
                Ra[i][j][k] = Ra[i][j][k]
    return Ra


def process_node(O):# 如何遍历 node 的结构, 备用函数
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
    mse = 0.0
    for i in range(label.shape[0]):
        score = mean_squared_error(label[i, 0, :], real[i, 0, :])
        mse += score
    return mse / label.shape[0]

# R2指的是相关系数，一般机器默认的是R2>0.99，这样才具有可行度和线性关系。
def r2(label, real):
    r2 = 0.0
    for i in range(label.shape[0]):
        score = r2_score(label[i, 0, :], real[i, 0, :])
        r2 += score
    return r2 / label.shape[0]

# 是用于度量两个变量X和Y之间的相关（线性相关），其值介于-1与1之间
# 计算出变量A和变量B的皮尔逊相关系数为0，不代表A和B之间没有相关性，只能说明A和B之间不存在线性相关关系。
def pear(label, real):
    p = 0.0
    for i in range(label.shape[0]):
        score = pearsonr(label[i, 0, :], real[i, 0, :])[0]
        p += score
    return p / label.shape[0]


def spear(label, real):
    sp = 0.0
    for i in range(label.shape[0]):
        score = spearmanr(label[i, 0, :], real[i, 0, :])[0]
        sp += score
    return sp / label.shape[0]

def prec(label, real):
    prec = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        precision = precision_score(labelz[i, 0, :], realz[i, 0, :]) # 取第0列
        prec += precision
    return prec/label.shape[0]

def recall(label, real):
    rec = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        recall = recall_score(labelz[i, 0, :], realz[i, 0, :]) # 取第0列
        rec += recall
    return rec/label.shape[0]

def f1(label, real):
    F1 = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        F1_score = f1_score(labelz[i, 0, :], realz[i, 0, :]) # 取第0列
        F1 += F1_score
    return F1/label.shape[0]

def AUC(label, real): #注意：auc需要比较label和输出的概率值
    auc = 0.0
    for i in range(label.shape[0]):
        arr_real = []
        arr_label = []
        auc = 0.0
        count = 0

        # 计算每一行最大值,也就是类别对应的分数
        for j in range(real.shape[2]): # 遍历每一行
            # 预测类别
            c = label[i,0,j]
            d = label[i,1,j]
            arr = np.array([c,d])
            label_index = np.argmax(arr) # 正类索引是0，负类索引是1，哪个维度是1，取哪个维度
            real_index = np.argmin(arr) # 0才是正类的索引，得到正类的score
            arr_label.append(label_index)

            # 预测score
            a = real[i,0,j]
            b = real[i,1,j]
            score_arr = [a,b]
            real_score = score_arr [real_index]
            arr_real.append(real_score)

        # 累加auc
        try:
            Auc_score = roc_auc_score(arr_label, arr_real)
        except:
            print('ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.')
        else:
            auc += Auc_score
            count += 1

    return auc/count

# def AUC(label, real): #注意：auc需要比较label和输出的概率值
#     auc = 0.0
#     count = 0
#     for i in range(label.shape[0]):
#         try:
#             Auc_score = roc_auc_score(label[i, 1, :], real[i, 1, :])  # 取第1列
#         except:
#             print('ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.')
#         else:
#             auc += Auc_score
#             count += 1
#     return auc/count