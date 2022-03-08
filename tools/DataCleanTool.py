# -*- coding: utf-8 -*-
# @Time : 2022/1/4 12:51
# @Author : CyrusMay WJ
# @FileName: DataCleanTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May
import os
import pandas as pd
from tools.PlotTool import PlotTool
from tools.utils import save_to_excel,decomp_by_tsne
import numpy as np
from sklearn.cluster import DBSCAN
from tools.StandardTool import StandardTool
import functools
import datetime
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import matplotlib.pyplot as plt


class CyrusPreprocessMessage():
    def __init__(self,message=None):
        self.message = message

    def __call__(self, func):
        @functools.wraps(func)
        def _deco(*args,**kwargs):
            print("*" * 20 + "start to use data_clean_tool in {}!".format(
                datetime.datetime.now().strftime("%Y-%m-%d")) + "*" * 20)
            return func(*args, **kwargs)
        return _deco


class BoxCleanNoise(object):
    def __init__(self,logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger)
    @CyrusPreprocessMessage("plot_box_dist")
    def plot_box_dist(self,data,columns=None):
        if not columns:
            if type(data) == pd.DataFrame:

                columns = data.columns.to_list()
                data = data.to_numpy()
            else:
                columns = ["x{}".format(i) for i in range(data.shape[1])]
        for i,path in zip([_ for _ in range(data.shape[1])],columns):
            path = path.replace("/","")
            self.plot_tool.boxplot(data[:,i],x_label=path,path=path)
            self.plot_tool.distplot(data[:,i],x_label=path,path=path)

    @CyrusPreprocessMessage("clean_box_noise_data")
    def clean_box_noise_data(self, data,delta = 3,columns=None):
        if not columns:
            if type(data) == pd.DataFrame:

                columns = data.columns.to_list()
                data = data.to_numpy()
            else:
                columns = ["x{}".format(i) for i in range(data.shape[1])]
        data = pd.DataFrame(data,columns=columns)
        for path in columns:
            self.logger.info("cleaning noise data of {} by box method".format(path))
            if path == "产品硫含量,%":
                continue
            q_down = data[path].quantile(q=0.25)
            q_up = data[path].quantile(q=0.75)
            down = q_down - (q_up-q_down) * delta
            up = q_up + (q_up-q_down) * delta
            data[path] = ((data[path] < down) * down) + ((data[path] >= down) * data[path])
            data[path] = ((data[path] > up) * up) + ((data[path] <= up) * data[path])
        save_to_excel(data,path="clean_box_noise_data")
        return data


class DBSCANCleanNoise():
    """
    detail description was shown in https://blog.csdn.net/Cyrus_May/article/details/113504879?spm=1001.2014.3001.5501
    """
    def __init__(self,logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger)




    def select_MinPts(self,data,k):
        # [b,sample,feature]
        standard_tool = StandardTool(x=data,logger=self.logger)
        data = standard_tool.transform_x(data)
        sample = np.expand_dims(data,axis=1)
        dists = ((sample - data)**2).sum(axis=-1)**0.5
        dists.sort(axis=1)
        dists = dists[:,k]
        dists.sort()
        dists = dists[::-1]
        save_to_excel(pd.DataFrame(dists[:,np.newaxis],columns=["{}_dist".format(k)]),path="{}_dist")
        self.plot_tool.lossplot(dists,path="{}_dist",x_label="样本编号",y_label="{}_dist")

    def dbscan_cluster(self,data,eps,k,path="dbscan"):
        standard_tool = StandardTool(x=data, logger=self.logger)
        data = standard_tool.transform_x(data)
        dbscan_model = DBSCAN(eps=eps, min_samples=k + 1)
        labels = dbscan_model.fit_predict(data)
        decom_data = decomp_by_tsne(data)
        self.plot_tool.dbscan_scatter_2d(decom_data,labels,path)
        save_to_excel(pd.DataFrame(np.concatenate([labels[:,None],decom_data],axis=1),columns=["x","y","label"]),path)


class LOFCleanNoise(object):
    def __init__(self,logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger)

    def cal_lof(self,data,k_ub=50,k_lb=10):
        if type(data) == pd.DataFrame:
            data = data.to_numpy()
        standard_tool = StandardTool(x=data,logger=self.logger)
        x_std = standard_tool.transform_x(data)
        result = []
        for k in range(k_lb,k_ub+1):
            lof_tool = LocalOutlierFactor(n_neighbors=k,p=2)
            lof_tool.fit(x_std)
            result.append(-lof_tool.negative_outlier_factor_)
        decom_data = decomp_by_tsne(x_std,n_dims=2)
        lof = np.array(result).max(axis=0)
        mask = (lof>1.5)*1
        self.plot_tool.dim_reduc_plot(x=decom_data[:,0],y = decom_data[:,1],classes=mask,x_label="x1",y_label="x2")
        data = pd.DataFrame(data)
        data["lof"] = lof
        save_to_excel(data,path="LocalOutlierFactor")
        lof.sort()
        self.plot_tool.lossplot(lof[::-1], path="LocalOutlierFactor", x_label="样本编号", y_label="LOF")
        return data


class Delta3CleanNoise(object):
    def __init__(self,logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger)

    @CyrusPreprocessMessage("3_delta_clean_data")
    def clean_data(self, data, columns=None):
        if not columns:
            if type(data) == pd.DataFrame:

                columns = data.columns.to_list()
                data = data.to_numpy()
            else:
                columns = ["x{}".format(i) for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=columns)

        ss_p_result = []
        for path in columns:
            self.logger.info("cleaning noise data of {} by 3_delta_clean_data method".format(path))
            tmp = data[path]
            tmp.index = [i for i in range(tmp.shape[0])]
            mean = tmp.mean()  # 计算均值
            std = tmp.std()  # 计算标准差

            ss,p = stats.kstest(tmp, 'norm', (mean, std))

            ss_p_result.append([ss,p])

            self.plot_tool.delta3plot(tmp,path,"概率密度")

        save_to_excel(pd.DataFrame(ss_p_result,columns=["统计量","p值"],index=columns), path="clean_box_noise_data")


