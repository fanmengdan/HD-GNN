# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl

# 测试 为每个relation 匹配 源端hunk 和 目的端hunk 的矩阵运算是否正确
if __name__ == '__main__':
    step = 2

    # x = np.load("./dataself/Cutting_Adjs/CAdjs_" + str(step) + ".npy", allow_pickle=True) # x是数据图 (100,40,40)
    IndexPaths_path = open(r'.\dataself\IndexPathList\IndexPathList_' + str(step) + '.pkl', 'rb')
    HunkIDmaps_path = open(r'.\dataself\HunkIDdict\HunkIDmap_' + str(step) + '.pkl', 'rb')
    IndexPaths = pkl.load(IndexPaths_path)
    HunkIDmaps = pkl.load(HunkIDmaps_path)

    RHr_data = np.zeros((100, 74, 89700), dtype=float); #(100, 74, 89700) 每个entity节点对的 源端hunk
    RHs_data = np.zeros((100, 74, 89700), dtype=float); #(100, 74, 89700) 每个entity节点对的 目的端hunk

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
                    Hr_key = indexLines[i].strip()
                    Hs_key = indexLines[j].strip()
                    if (Hr_key != 'null'):
                        Hr_num = HunkIDmap[Hr_key]
                        if(Hr_num < 74): # 限制hunkID 标号大小
                            RHr_data[index1, Hr_num, cnt1] = 1.0
                    if (Hs_key != 'null'):
                        Hs_num = HunkIDmap[Hs_key]
                        if (Hs_num < 74):
                            RHs_data[index1, Hs_num, cnt1] = 1.0
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
    #                 Hr_key = indexLines[i].strip()
    #                 Hs_key = indexLines[j].strip()
    #                 if (Hr_key != 'null'):
    #                     Hr_num = HunkIDmap[Hr_key]
    #                     RHr_data[index1, Hr_num, cnt1] = 1.0
    #                 if (Hs_key != 'null'):
    #                     Hs_num = HunkIDmap[Hs_key]
    #                     RHs_data[index1, Hs_num, cnt1] = 1.0
    #         cnt1 += 1;

    print("")