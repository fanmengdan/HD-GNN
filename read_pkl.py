import joblib
import numpy as np
step = 2

# rb是2进制编码文件，文本文件用r
# IndexPathList = joblib.load(open(r'Adjset/glide/IndexPathList/IndexPathList_' + str(step) + '.pkl','rb'))
# HunkIDmap = joblib.load(open(r'Adjset/glide/HunkIDdict/HunkIDmap_' + str(step) + '.pkl','rb'))

# E_node_train = joblib.load(open(r'./Intermediate_products/glide/E_node_train_' + str(step) + '.pkl', 'rb'))
# E_node_test = joblib.load(open(r'./Intermediate_products/glide/E_node_test_' + str(step) + '.pkl', 'rb'))

# E_edge_train = joblib.load(open(r'./Intermediate_products/glide/E_edge_train_' + str(step) + '.pkl', 'rb'))
# E_edge_test = joblib.load(open(r'./Intermediate_products/glide/E_edge_test_' + str(step) + '.pkl', 'rb'))

# C_edge_train = joblib.load(open(r'./Intermediate_products/glide/C_edge_train_' + str(step) + '.pkl', 'rb'))
# C_edge_test = joblib.load(open(r'./Intermediate_products/glide/C_edge_test_' + str(step) + '.pkl', 'rb'))

# Es_data = joblib.load(open(r'./Intermediate_products/glide/Es_data_' + str(step) + '.pkl', 'rb'))
# Et_data = joblib.load(open(r'./Intermediate_products/glide/Et_data_' + str(step) + '.pkl', 'rb'))

Hr_label = joblib.load(open(r'./Intermediate_products/glide/Hr_label_' + str(step) + '.pkl', 'rb'))
Ct_label = joblib.load(open(r'./Intermediate_products/glide/Ct_label_' + str(step) + '.pkl', 'rb'))

Esc_data = joblib.load(open(r'./Intermediate_products/glide/Esc_data_' + str(step) + '.pkl', 'rb'))
Etc_data = joblib.load(open(r'./Intermediate_products/glide/Etc_data_' + str(step) + '.pkl', 'rb'))

print('done')

# glide, step = 2
# parser.add_argument('--No', type=int, default=200, help='The Number of Objects') # 200, 250, 250
# parser.add_argument('--HNo', type=int, default=74, help='The Number of Objects') # 74, 114, 150
# parser.add_argument('--Nr', type=int, default=39800, help='The Number of Relations') # 39800, 62250, 62250
# parser.add_argument('--HNr', type=int, default=5402, help='The Number of Relations') # 5402, 12882, 22350

# glide, step = 3
# parser.add_argument('--No', type=int, default=250, help='The Number of Objects') # 200, 250, 250
# parser.add_argument('--HNo', type=int, default=114, help='The Number of Objects') # 74, 114, 150
# parser.add_argument('--Nr', type=int, default=62250, help='The Number of Relations') # 39800, 62250, 62250
# parser.add_argument('--HNr', type=int, default=12882, help='The Number of Relations') # 5402, 12882, 22350

# glide, step = 5
# parser.add_argument('--No', type=int, default=250, help='The Number of Objects') # 200, 250, 250
# parser.add_argument('--HNo', type=int, default=150, help='The Number of Objects') # 74, 114, 150
# parser.add_argument('--Nr', type=int, default=62250, help='The Number of Relations') # 39800, 62250, 62250
# parser.add_argument('--HNr', type=int, default=22350, help='The Number of Relations') # 5402, 12882, 22350