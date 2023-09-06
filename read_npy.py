import numpy as np


step = 2
# input
x = np.load("./dataself/Adj/Adjs_" + str(step) + ".npy", allow_pickle=True)
y = np.load("./dataself/Adj/HunkAdjs_" + str(step) + ".npy", allow_pickle=True)

x1 = np.load("./dataself/Padding_Adjs/PAdjs_" + str(step) + ".npy", allow_pickle=True)
y1 = np.load("./dataself/Padding_Adjs/PHunkAdjs_" + str(step) + ".npy", allow_pickle=True)

x2 = np.load("./dataself/Cutting_Adjs/CAdjs_" + str(step) + ".npy", allow_pickle=True)
y2 = np.load("./dataself/Cutting_Adjs/CHunkAdjs_" + str(step) + ".npy", allow_pickle=True)


print(x)
