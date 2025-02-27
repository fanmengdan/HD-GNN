 <div align="center">
  <h1 id="HD-GNN"><a href="https://gitee.com/fanmengdan1/fgfe/" target="repo">HD-GNN</a></h1>
</div>

#### For each commit, given its adjacency matrixs of entity reference graph and code change graph (Preprocessing code in https://github.com/fanmengdan/Preprocess_graph), this code aims at using HD-GNN to train a model which can classify the relations among code changes. 
#### Due to the large size of the data file, we only present the preprocessed dataset of glide here(glide.zip needs to be decompressed). Please create other data according to https://github.com/fanmengdan/Preprocess_graph

## Requirements
- macOS/Windows/Linux
- anaconda (python 3.7)

## Usage

### 1. Load adjacency matrixs of  entity reference graph and code change graph of each commit.

 - Enter adjacency matrixs data path in ***utils2.py***：
```sh
repo = 'glide'
x = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)  # x是entity adj (100,200,200)  
y = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)  # y是hunk adj (100,74,74)  
```
 - Enter some file paths that can assist in generating intermediate products that HD-GNN can learn from  in ***utils2.py***：
 ```sh
 IndexPathList = open(r'./dataset/'+repo+'/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')  
HunkIDmaps_path = open(r'./dataset/'+repo+'/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')  
IndexPaths = joblib.load(IndexPathList)  
HunkIDmaps = joblib.load(HunkIDmaps_path)
```

### 2. Train and Test
 - Set model parameters in ***main.py***, especially:
 ```sh
parser.add_argument('--Ne', type=int, default=entity_node, help='The Number of entities')  # 200, 250, 250  
parser.add_argument('--Nc', type=int, default=hunk_node, help='The Number of code changes')  # 74, 114, 150  
parser.add_argument('--Ner', type=int, default=entity_edge,  
  help='The Number of entity Relations')  # 39800, 62250, 62250  
parser.add_argument('--Ncr', type=int, default=hunk_edge,  
  help='The Number of code change Relations')  # 5402, 12882, 22350  
parser.add_argument('--Step', type=int, default=step, help='the number of commits/groups')  # 2, 3, 5  
parser.add_argument('--Repo', type=str, default='glide', help='the name of repository')
```
 - Set up training or testing in ***main.py***:
 ```sh
 parser.add_argument('--Type', dest='Type', default='train', help='train or test')
 ```
Then run the ***main.py***.
