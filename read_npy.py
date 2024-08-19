import numpy as np

step = 2
No = 200
repo = 'glide'

# input
Adjs = np.load('./Adjset/'+repo+'/Adj/Adjs_' + str(step) + '.npy', allow_pickle=True)
HunkAdjs = np.load('./Adjset/'+repo+'/Adj/HunkAdjs_' + str(step) + '.npy', allow_pickle=True)

PAdjs = np.load('./Adjset/'+repo+'/Padding_Adjs/PAdjs_' + str(step) + '.npy', allow_pickle=True)
PHunkAdjs = np.load('./Adjset/'+repo+'/Padding_Adjs/PHunkAdjs_' + str(step) + '.npy', allow_pickle=True)

CAdjs = np.load('./Adjset/'+repo+'/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)
CHunkAdjs = np.load('./Adjset/'+repo+'/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)

# output
# HRa_t = np.load('outputSelf/glide/model_4/' + str(step) + '/HRa_t' + str(No) + '.npy', allow_pickle=True)
# HRa_y = np.load('outputSelf/glide/model_4/' + str(step) + '/HRa_y' + str(No) + '.npy', allow_pickle=True)

print('done')