import joblib
import numpy as np
step = 2

# rb是2进制编码文件，文本文件用r
f5 = open(r'.\dataself\rocketmq\IndexPathList\IndexPathList_' + str(step) + '.pkl','rb')
f8 = open(r'.\dataself\rocketmq\HunkIDdict\HunkIDmap_' + str(step) + '.pkl','rb')
Ra_data_train = joblib.load(open(r'.\Intermediate_products\rocketmq\Ra_data_train_' + str(step) + '.pkl', 'rb'))
Ra_label_train = joblib.load(open(r'.\Intermediate_products\rocketmq\Ra_label_train_' + str(step) + '.pkl', 'rb'))

data5 = joblib.load(f5)
data8 = joblib.load(f8)

print(data8)