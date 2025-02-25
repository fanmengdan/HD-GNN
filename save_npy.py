import numpy as np
import pandas as pd

data = pd.read_csv('./data/ER_40/ER-40-target-1.csv')
data2 = np.array(data)

# if __name__ == '__main__':
#     with open('./data/IoT_20/IoT-20-input-0.0-0.2-0.2-296.csv', 'r') as fp:
#         reader = csv.reader(fp)
#
#         # titles = next(reader)  # 剪切reader第一行的值返回给title
#         # print(titles)
#         # print("***************")
#         # for x in reader:
#         #     print(x)
np.save('./data/npy/erdos40_y.npy', data)



