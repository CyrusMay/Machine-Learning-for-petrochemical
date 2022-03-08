# -*- coding: utf-8 -*-
# @Time : 2022/1/5 14:07
# @Author : CyrusMay WJ
# @FileName: MetricsTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May
from sklearn.metrics import calinski_harabasz_score,silhouette_score,silhouette_samples
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error,explained_variance_score,median_absolute_error
import pandas as pd
import numpy as np
from tools.utils import save_to_excel
from sklearn.metrics import confusion_matrix,classification_report,auc,precision_score
import datetime

class CyrusMetrics():
    def __init__(self,logger=None):
        self.logger = logger

    def classfication_metrics(self,y,y_pre,y_prob):
        conf = confusion_matrix(y,y_pre)
        scores = classification_report(y,y_pre)
        auc_val = auc(y,y_prob)
        print(auc_val)
        print(scores)
        print(conf)
        print(precision_score(y,y_pre))
        # save_to_excel([(pd.DataFrame(conf),"confusion_matrix"),
        #                (pd.DataFrame(scores),"classification_report"),
        #                (pd.DataFrame([[auc_val]]),"auc")],path="class")

    def cluster_metrics(self,data,label):
        # 1、CH分数
        ch_score = calinski_harabasz_score(data,label)
        # 2. 轮廓系数
        ss_score = silhouette_score(data,label)
        # 3、样本轮廓系数
        ss_sample = silhouette_samples(data, label)
        # 4. SSE
        data = pd.DataFrame(data)
        data["label"] = label
        label = set(label)
        sse = 0
        for idx in label:
            tmp = data[data["label"] == idx].iloc[:,:-1].to_numpy()
            center = tmp.mean(axis=0)
            sse += ((tmp - center)**2).sum()
        sse_score = sse
        return pd.DataFrame([[ch_score,ss_score,sse_score]],columns=["CH分数","轮廓系数","SSE"]),pd.DataFrame(ss_sample[:,None],columns=["样本轮廓系数"])

    def regression_metrics(self,y,y_pre,path=None):
        if type(y) == pd.DataFrame:
            y = y.to_numpy()
        if type(y_pre) == pd.DataFrame:
            y_pre = y_pre.to_numpy()
        data_result = pd.DataFrame()

        result = pd.DataFrame()
        if len(y.shape) > 1:
            if y.shape[1] > 1:
                for i in range(y.shape[1]):
                    result["y{}".format(i+1)] = self.__sub_regression(y[:,i],y_pre[:,i])
                    data_result["y{}真实值".format(i+1)] = y[:,i]
                    data_result["y{}预测值".format(i + 1)] = y_pre[:, i]
                save_to_excel([(result,"评估指标"),(data_result,"预测值")], path=path + "_metrics_"+datetime.datetime.now().strftime("%Y-%m-%d"))
                return result
            else:
                y = y[:,0]
                y_pre = y_pre if len(y_pre.shape) == 1 else y_pre[:,0]
        metric = self.__sub_regression(y,y_pre)
        result["y"] = metric
        data_result["实际值"] = y
        data_result["预测值"] = y_pre
        save_to_excel([(result,"评估指标"),(data_result,"预测值")], path=path + "_metrics_"+datetime.datetime.now().strftime("%Y-%m-%d"))
        return result

    def __sub_regression(self,y,y_pre):
        mae = mean_absolute_error(y,y_pre)
        mse = mean_squared_error(y,y_pre)
        r2 = r2_score(y,y_pre)
        mape = mean_absolute_percentage_error(y,y_pre)
        ex_var = explained_variance_score(y,y_pre)
        medae = median_absolute_error(y,y_pre)
        return pd.Series([mae,mse,r2,mape,ex_var,medae],\
                     index=["平均绝对误差","均方误差","决定系数","平均绝对百分比误差","解释方差","中位数绝对误差"])



if __name__ == '__main__':

    x = np.random.randn(100,10)
    y = np.random.randint(0,3,100)

    metric = CyrusMetrics()
    metric.cluster_metrics(x,y)







