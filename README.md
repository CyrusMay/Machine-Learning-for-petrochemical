# Machine-Learning-for-petrochemical
Provide a complete set of petrochemical modeling process


## 1.Use in clustering
```
import logging
import sys
import pandas as pd
import numpy as np
from tools.ClusterTool import K_meansCluster
from tools.utils import save_to_excel

np.random.seed(2022)
tf.random.set_seed(2022)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)


def run(op_state = 1):

    # read data
    data = pd.DataFrame(np.random.rand(200,10))
    
    # select best k for K-Means
    if op_state == 1:
        cluster_tool = K_meansCluster(logger)
        cluster_tool.select_best_k(data)
        
    # cluster, plot, save for K-Means
    elif op_state == 2:
        cluster_tool = K_meansCluster(logger)
        labels = cluster_tool.run_cluster(data,n_clusters=3)
        cluster_tool.plot_2d(data,label=labels)
        cluster_tool.plot_3d(data,label=labels)
        cluster_tool.plot_hist(data,label=labels)
        cluster_tool.cal_center(data,label=labels)
        data["label"] = labels
        save_to_excel(data, "kmeans_result")

if __name__ == '__main__':
    op_state = 1
    run(op_state)

```
## 2. Use in data cleaning
```
import logging

import sys
from tools_old.DBTool import DBTool
import pandas as pd
import pickle as pkl
from tools_old.MergeData import MergeData
from tools_old.PreprocessTool import PreprocessData,DBSCANCleanNoise
from tools.DataCleanTool import LOFCleanNoise,BoxCleanNoise
from  tools.FeatureTool import DataAnalysis
from tools.utils import save_to_excel

logger = logging.getLogger(name="cyrus_")
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

def run():

    # read data
    data = pd.DataFrame(np.random.rand(200,10))
    
    # clean data by DBSCAN
    if op_state == 1:
        dbscan_tool = DBSCANCleanNoise(logger)
        # select EPS for DBSCAN
        dbscan_tool.select_MinPts(data,k=4)
        dbscan_tool.dbscan_cluster(prop,eps=0.1,k=5)
        
    # clean data by Box-Plot
    elif op_state ==2:
        box_tool = BoxCleanNoise(logger)
        data_col = box_tool.clean_box_noise_data(data)
        save_to_excel(data_col, path="preprocess_data")
        
    # clean data by LOF
    elif op_state ==3:
        lof_tool = LOFCleanNoise(logger)
        data_col = lof_tool.cal_lof(data)
        data_col = data_col[data_col["lof"] < 1.5]
        save_to_excel(data_col, path="preprocess_data")
    
    # calculate distribution index
    elif op_state == 4:
        data_tool = DataAnalysis(logger)
        data_tool.cal_distribution(data)

if __name__ == '__main__':
    global op_state
    op_state = 1
    run()
```
## 3. Use in feature selection
```
import logging
import sys
from  tools.FeatureTool import CorrelationsTool
from tools.utils import save_to_excel
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)


def run():
    x = pd.DataFrame(np.random.rand(200,10))
    y = pd.DataFrame(np.random.rand(200,1))

    corr_tool = CorrelationsTool(logger)
    
    # calculate maximal information coefficient (MIC)
    corr_tool.mic(x=x,y=y)
    
    # calculate Pearson Coefficient
    corr_tool.pearson(x)
    

if __name__ == '__main__':
    run()

```

## 4. Use in modeling

```
import logging
import sys
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
from tools.utils import save_var
from tools.DBTool import DBTool
from sklearn.model_selection import train_test_split
from tools.ModelTool import CyrusSVR,CyrusLR,CyrusXGB,CyrusNN,CyrusCNSS
import tensorflow as tf
tf.random.set_seed(2022)
np.random.seed(2022)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

x = np.random.rand(200,10)
y = np.random.rand(200,1)

# build SVR model
svr = CyrusSVR(x=x1_train,y=y1_ron_train,logger=logger)
# select hyper parameters
svr.hyper_paras_select(init_params = {'kernel': 'rbf', 'gamma': 0.01, 'C': 5, 'epsilon': 0.1, }, 
                      cv_params = {'epsilon': [0.01, 0.05, 0.1,0.2, 0.5,0.8, 1]})
# train model
svr.build_model(gamma=0.01,C=5,epsilon=0.1)
svr.fit()
# evaluate and visualization
svr.evaluate(x_test=x1_test,y_test=y1_ron_test)
svr.plot_pre(x_test=x1_test,y_test=y1_ron_test)

# build neural network
nn = CyrusNN(x=x1_train,y=y1_ron_train,logger=logger)
nn.build_model(net_structe=[21,15,1])
nn.fit(epoch=150,val_data=(x1_test,y1_ron_test))
nn.evaluate(x_test=x1_test, y_test=y1_ron_test)
nn.plot_pre(x_test=x1_test, y_test=y1_ron_test)
```
## 5. Optimalization of process parameters
```
import logging
import sys
import pandas as pd
import numpy as np
from tools.utils import save_var
from tools.OptTool import GAOptimizer
import tensorflow as tf
tf.random.set_seed(2022)
np.random.seed(2022)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)


opt_tool = GAOptimizer(model_path="./best_model/NN.h5",standard_tool_path="./best_model/standard_tool.pkl",
                              is_min=False,var_is_opt = [False]*9+[True]*12,
                              lb=[],
                              up=[])


opt_tool.build_opt(max_iter=200)
x = np.array([])
tmp = opt_tool.run(preopt_var=x)
```
