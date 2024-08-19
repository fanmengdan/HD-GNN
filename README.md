
 <div align="center">  
  <h1 id="HD-GNN"><a href="https://gitee.com/fanmengdan1/fgfe/" target="repo">HD-GNN</a></h1>  
</div>  
  
#### For each commit, given its adjacency matrixs of  entity reference graph and code change graph , this code aims at using HD-GNN to train a model which can classify  the relations among code changes.  
  
## Requirements  
 - macOS/Windows/Linux  
 - anaconda (python 3.7)  
  
## Dataset  
The dataset is in **HD-GNN** folder, which is the information of entity reference graph and code change graph of each tangled commit.  As described in section 5.2 of the paper, we used a common dataset. Each composite/tangled commit in this dataset is composed of 2/3/5 atomic commits.  Directory *HD-GNN/dataset/glide/2/*, *HD-GNN/dataset/glide/3/* and *HD-GNN/dataset/glide/5/* are the tangled commits each of which consists of 2, 3 and 5 atomic commits, respectively. Each subfolder, such as the folder *0bdf6a7_928c9a1* in the folder *HD-GNN/dataset/glide/5/*, contains the ground truth files of the tangled commit consisting of commits 0bdf6a7 and 928c9a1. For example, in folder 0bdf6a7_928c9a1, there are following files:

 - **GT\_0bdf6a7\_928c9a1.txt**  # The ground truth. It records the code
   change included by each atomic commit using their IDs.
 - **Id\_0bdf6a7\_928c9a1.txt**  # The entity ID information. It records
   entity of the tangled commit using their IDs.
 - **Type\_0bdf6a7\_928c9a1.txt** # The entity type information. It records
   type of each entity in the tangled commit.
 - **Index\_0bdf6a7\_928c9a1.txt** # The index information. It records each
   entity in which code changes using their IDs.
 - **Source\_0bdf6a7\_928c9a1.txt** # The syntactic relation information.
   It records the source entity of each syntactic relation.
 - **Target\_0bdf6a7\_928c9a1.txt** # The syntactic relation information. It
   records the target entity of each syntactic relation.

  
## Usage  
  
### 1. Load adjacency matrixs of  entity reference graph and code change graph of each commit.  
  
 - Enter adjacency matrixs data path in ***utils1.py***：  
```sh  
repo = 'glide'  
x = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)  # x是entity adj (100,200,200) y = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)  # y是hunk adj (100,74,74)   
```  
 - Enter some file paths that can assist in generating intermediate products that HD-GNN can learn from  in ***utils1.py***：  
 ```sh  
 IndexPathList = open(r'./dataset/'+repo+'/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb') HunkIDmaps_path = open(r'./dataset/'+repo+'/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb') IndexPaths = joblib.load(IndexPathList) HunkIDmaps = joblib.load(HunkIDmaps_path)  
```  
  
### 2. Train and Test  
 - Set model parameters in ***main.py***, especially:  
 ```sh  
parser.add_argument('--Ne', type=int, default=entity_node, help='The Number of entities')  # 200, 250, 250 parser.add_argument('--Nc', type=int, default=hunk_node, help='The Number of code changes')  # 74, 114, 150 parser.add_argument('--Ner', type=int, default=entity_edge,    
help='The Number of entity Relations')  # 39800, 62250, 62250 parser.add_argument('--Ncr', type=int, default=hunk_edge,    
help='The Number of code change Relations')  # 5402, 12882, 22350 parser.add_argument('--Step', type=int, default=step, help='the number of commits/groups')  # 2, 3, 5 parser.add_argument('--Repo', type=str, default='glide', help='the name of repository')  
```  
 - Set up training or testing in ***main.py***:  
 ```sh  
 parser.add_argument('--Type', dest='Type', default='train', help='train or test')  
 ```Then run the ***main.py***.