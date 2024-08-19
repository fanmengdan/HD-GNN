# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl

# 测试 为每个relation 匹配 源端hunk 和 目的端hunk 的矩阵运算是否正确
if __name__ == '__main__':
    step = 2

    # x = np.load("./Adjset/Cutting_Adjs/CAdjs_" + str(step) + ".npy", allow_pickle=True) # x是数据图 (100,40,40)
    IndexPatCt_path = open(r'./Adjset/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')
    HunkIDmaps_path = open(r'./Adjset/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')
    IndexPaths = pkl.load(IndexPatCt_path)
    HunkIDmaps = pkl.load(HunkIDmaps_path)

    Esc_data = np.zeros((100, 74, 89700), dtype=float); #(100, 74, 89700) 每个entity节点对的 源端hunk
    Etc_data = np.zeros((100, 74, 89700), dtype=float); #(100, 74, 89700) 每个entity节点对的 目的端hunk

    # cutting 为每个relation 匹配 源端hunk 和 目的端hunk
    for index1 in range(100):  # 遍历100个图，每个图都有一个map
        # 识别i和j分别在哪个hunk里
        IndexPath = IndexPaths[index1]
        HunkIDmap = HunkIDmaps[index1]  # 是一个dict

        # 读取index
        index_txt = open(IndexPath)
        indexLines = index_txt.readlines()[:300]  # 按行读取，限制entity 标号大小
        # 分配源端、目的端 hunkIDnum
        cnt1 = 0
        for i in range(len(indexLines)): # 有的entity图可能比(40,40)小，也有可能大
            for j in range(len(indexLines)):
                if (i != j):
                    Cs_key = indexLines[i].strip()
                    Ct_key = indexLines[j].strip()
                    if (Cs_key != 'null'):
                        Cs_num = HunkIDmap[Cs_key]
                        if(Cs_num < 74): # 限制hunkID 标号大小
                            Esc_data[index1, Cs_num, cnt1] = 1.0
                    if (Ct_key != 'null'):
                        Ct_num = HunkIDmap[Ct_key]
                        if (Ct_num < 74):
                            Etc_data[index1, Ct_num, cnt1] = 1.0
                    cnt1 += 1;


    # # padding 为每个relation 匹配 源端hunk 和 目的端hunk
    # for index1 in range(100):  # 遍历100个图，每个图都有一个map
    #     # 识别i和j分别在哪个hunk里
    #     IndexPath = IndexPaths[index1]
    #     HunkIDmap = HunkIDmaps[index1]  # 是一个dict
    #
    #     # 读取index
    #     index_txt = open(IndexPath)
    #     indexLines = index_txt.readlines() # 按行读取，限制entity 标号大小
    #     # 分配源端、目的端 hunkIDnum
    #     cnt1 = 0
    #     for i in range(len(indexLines)): # 有的entity图可能比(40,40)小，也有可能大
    #         for j in range(len(indexLines)):
    #             if (i != j):
    #                 Cs_key = indexLines[i].strip()
    #                 Ct_key = indexLines[j].strip()
    #                 if (Cs_key != 'null'):
    #                     Cs_num = HunkIDmap[Cs_key]
    #                     Esc_data[index1, Cs_num, cnt1] = 1.0
    #                 if (Ct_key != 'null'):
    #                     Ct_num = HunkIDmap[Ct_key]
    #                     Etc_data[index1, Ct_num, cnt1] = 1.0
    #         cnt1 += 1;

    print("")