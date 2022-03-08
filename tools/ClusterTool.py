# -*- coding: utf-8 -*-
# @Time : 2022/1/5 15:28
# @Author : CyrusMay WJ
# @FileName: SystemCluster.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May

from sklearn.cluster import AgglomerativeClustering,KMeans
from scipy.cluster.hierarchy import dendrogram,linkage
from tools.PlotTool import PlotTool
from tools.MetricsTool import CyrusMetrics
from tools.StandardTool import StandardTool
from tools.utils import save_to_excel
import pandas as pd
import math
import numpy as np



class CyrusCluster():
    def __init__(self,logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger)

    def plot_2d(self,data,label,path=None):
        assert type(data) == pd.DataFrame,"请传入DataFrame格式的数据！"
        columns = data.columns.to_list()
        cnt = 1
        for i in range(len(columns)):
            for j in range(i+1,len(columns)):
                self.logger.info("plotting 2d cluster scatter: {}".format(cnt))
                self.plot_tool.cluster_scatter_2d(data[columns[i]],data[columns[j]],classes=label,\
                                                  path=(columns[i]+"_"+columns[j]).replace("/",""),x_label=columns[i],\
                                                  y_label=columns[j])
                cnt += 1
    def plot_3d(self,data,label,path=None):
        assert type(data) == pd.DataFrame, "请传入DataFrame格式的数据！"
        columns = data.columns.to_list()
        cnt = 1
        for i in range(len(columns)):
            for j in range(i+1,len(columns)):
                for k in range(j + 1, len(columns)):
                    self.logger.info("plotting 3d cluster scatter: {}".format(cnt))
                    x,y,z = data[columns[i]],data[columns[j]],data[columns[k]]
                    self.plot_tool.cluster_scatter_3d(x, y, z,classes=label, x_label=columns[i], y_label=columns[j], \
                                                      z_label=columns[k],path=(columns[i]+"_"+columns[j]+"_"+columns[k]))
                    cnt += 1

    def plot_hist(self,data,label,path=None):
        assert type(data) == pd.DataFrame, "请传入DataFrame格式的数据！"
        columns = data.columns.to_list()
        data["label"] = label
        label = list(set(label))
        cnt = 1
        for col in columns:
            self.logger.info("plotting cluster of multi-hist: {}".format(cnt))
            tmp = []
            for i in label:
                tmp.append(data[data["label"]==i][col].to_list())
            self.plot_tool.multi_hist(tmp,x_label=col,y_label="频数",path=col)
            cnt += 1

    def metrics_summary(self,data,label,path="k_means"):
        standard_tool = StandardTool(data,self.logger)
        data = standard_tool.transform_x(data)
        metrics_tool = CyrusMetrics(self.logger)
        metrics,ss_sample = metrics_tool.cluster_metrics(data,label)
        self.plot_tool.silhouette_plot(silhouette_sample=ss_sample,labels=label,path=path)
        save_to_excel(metrics,path="metrics_" +path)
        self.logger.info("metrics for {} has been saved!".format(path))
        save_to_excel(ss_sample,path= "ss_sample_" + path)
        self.logger.info("silhouette_samples for {} has been saved!".format(path))

    def cal_center(self,data,label,path="聚类中心"):
        assert type(data) == pd.DataFrame,"请传入DataFrame！"
        idxs = np.unique(label)
        idxs.sort()
        columns = data.columns
        data = data.to_numpy()
        result = []
        for idx in idxs:
            result.append(data[label==idx].mean(axis=0))
        save_to_excel(pd.DataFrame(result,columns=columns),path)
        self.logger.info(pd.DataFrame(result,columns=columns))


    def select_best_k(self,data,path="k_means",dist_method="ward"):
        down = 2
        up = math.ceil(data.shape[0] ** 0.5)
        up = 10
        SSE_all = []
        lunkuoxishu_all = []
        ch_score = []
        self.logger.info("start to select best k for k_means!")
        standard_tool = StandardTool(data, self.logger)
        std_data = standard_tool.transform_x(data)
        metrics_tool = CyrusMetrics(self.logger)

        for n in range(down, up + 1):
            self.logger.info("正在进行{}个聚类中心计算".format(n))
            if path == "k_means":
                labels = self.run_cluster(data, n_clusters = n,select_k=True)
            else:
                labels = self.run_cluster(data, n_clusters=n, select_k=True,dist_method=dist_method)
            metrics, ss_sample = metrics_tool.cluster_metrics(std_data, labels)
            SSE_all.append(metrics["SSE"].iloc[0])
            lunkuoxishu_all.append(metrics["轮廓系数"].iloc[0])
            ch_score.append(metrics["CH分数"].iloc[0])

        self.plot_tool.line_scatter(x=[i for i in range(down, up + 1)], y=SSE_all, x_label="聚类数目", y_label="SSE",path="{}_sse".format(path))
        self.plot_tool.line_scatter(x=[i for i in range(down, up + 1)], y=lunkuoxishu_all, x_label="聚类数目", y_label="轮廓系数",
                                    path="{}_轮廓系数".format(path))
        self.plot_tool.line_scatter(x=[i for i in range(down, up + 1)], y=ch_score, x_label="聚类数目", y_label="CH分数",
                                    path="{}_CH".format(path))

        save_to_excel(pd.DataFrame(np.array([[i for i in range(down, up + 1)], SSE_all, lunkuoxishu_all,ch_score]).T,
                         columns=["聚类中心数目", "SSE", "轮廓系数","CH分数"]),path="{}聚类中心选取".format(path))


class SystemCluster(CyrusCluster):
    def __init__(self,logger=None,**params):
        super(SystemCluster,self).__init__(logger,**params)

    def run_cluster(self,data,n_clusters=2,dist_method = 'ward',select_k=False):
        """
        :param n_clusters:
        :param dist_method:  ward or average
        :return:
        """
        standard = StandardTool(x=data,logger=self.logger)
        stand_data = standard.transform_x(data)
        self.obj = AgglomerativeClustering(n_clusters=n_clusters,linkage=dist_method)
        labels = self.obj.fit_predict(stand_data)
        if not select_k:
            self.plot_tool.tree_image(stand_data,dist_method)
            self.logger.info("tree_image for SystemCluster has been plotted!")
            self.metrics_summary(data, labels, path="SystemCluster")

        return labels

class K_meansCluster(CyrusCluster):
    def __init__(self,logger=None,**params):
        super(K_meansCluster,self).__init__(logger,**params)

    def run_cluster(self,data,n_clusters=2,select_k=False):
        """
        :param n_clusters:
        :param
        :return:
        """
        standard = StandardTool(x=data,logger=self.logger)
        stand_data = standard.transform_x(data)
        self.obj = KMeans(n_clusters=n_clusters)
        labels = self.obj.fit_predict(stand_data)
        if not select_k:
            self.metrics_summary(data, labels, path="KmeansCluster")
        return labels
