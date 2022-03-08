# Machine-Learning-for-petrochemical
Provide a complete set of petrochemical modeling process

## 1.Use in clustering
```
from tools.ClusterTool import K_meansCluster

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

```
