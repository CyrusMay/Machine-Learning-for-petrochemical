# -*- coding: utf-8 -*-
# @Time : 2022/1/4 13:14
# @Author : CyrusMay WJ
# @FileName: PlotTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import numpy as np
import datetime
import matplotlib
import pandas as pd
import logging
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram,linkage
import functools
import os
from statsmodels.graphics.api import qqplot


class CyrusPlot():
    def __init__(self):
        pass
    def __call__(self,func):
        @functools.wraps(func)
        def _deco(*args,**kwargs):
            print("*"*20+"start to use plot_tool in {}!".format(datetime.datetime.now().strftime("%Y-%m-%d"))+"*"*20)
            return func(*args,**kwargs)
        return _deco


class PlotTool():
    def __init__(self,logger=None):
        if not logger:
            logger = logging.getLogger(name = "plot_tool")
            logger.setLevel(logging.INFO)
            screen_handler = logging.StreamHandler(sys.stdout)
            screen_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
            screen_handler.setFormatter(formatter)
            logger.addHandler(screen_handler)
        self.logger = logger
        if "image" not in os.listdir():
            os.mkdir("./image")
        plt.rcParams['axes.unicode_minus'] = False
        self.color = ["g", "r", "b", "y", "c", "m","b"]
        self.marker = ["o", "*", "s", "+", "x", "v","<"]

    def font(self,size = 30):
        return FontProperties(fname = "C:\Windows\Fonts\simsun.ttc",size=size)

    @CyrusPlot()
    def QQPlot(self,x,label="x",path = None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")

        x = np.array(x).reshape([-1])
        qqplot(x,marker="+",line="s")
        plt.xticks(fontproperties=self.font(10))
        plt.yticks(fontproperties=self.font(10))
        plt.xlabel("标准正态分布分位数", fontproperties=self.font(15))
        plt.ylabel("{}样本的分位数".format(label), fontproperties=self.font(15))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        # plt.tight_layout()
        # x_min, x_max = np.array(x).min(), np.array(x).max()
        # y_min, y_max = np.array(x).min(), np.array(x).max()
        # delta_x = x_max - x_min
        # delta_y = y_max - y_min
        # x_min, x_max = x_min - 0.05 * delta_x, x_max + 0.05 * delta_x
        # y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        # plt.plot([x_min,x_max],[y_min,y_max],color="r",linestyle="--",linewidth=1)
        # plt.xlim((x_min, x_max))
        # plt.ylim((y_min, y_max))
        plt.savefig("./image/qq_{}".format(label) + path + ".png")
        # plt.show()
        plt.close()

    @CyrusPlot()
    def dim_reduc_plot(self,x,y,classes,x_label="x",y_label="y",path = None,legend = ["非异常值","异常值"]):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        x = np.array(x).reshape([-1,1])
        y = np.array(y).reshape([-1, 1])
        data = np.array(np.concatenate([x,y],axis=-1))
        data = pd.DataFrame(data,columns=["x_{}".format(i) for i in range(1,data.shape[1]+1)])
        data["label"] = classes
        idx = [0,1]


        figure = plt.figure(figsize=[30, 16], dpi=72)
        for count,i in enumerate(idx):
            tmp = data[data["label"] == i]
            x,y = tmp.iloc[:,0],tmp.iloc[:,1]
            plt.scatter(x,y,c=self.color[count],marker=self.marker[count],s=150,label=legend[count])

        x,y = data.iloc[:,0],data.iloc[:,1]
        x_min, x_max = np.array(x).min(), np.array(x).max()
        y_min, y_max = np.array(y).min(), np.array(y).max()
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        x_min, x_max = x_min - 0.05 * delta_x, x_max + 0.05 * delta_x
        y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.xlabel(x_label, fontproperties=self.font(45))
        plt.ylabel(y_label, fontproperties=self.font(45))
        plt.legend(prop =self.font(45))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/dim_reduc_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def delta3plot(self, data, x_label, y_label,  path=""):
        mean = data.mean()  # 计算均值
        std = data.std()  # 计算标准差


        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        data.plot(kind='kde', grid=True, style='-k')
        plt.axvline(3 * std + mean, color='r', linestyle="--", alpha=0.8)
        plt.axvline(-3 * std + mean, color='r', linestyle="--", alpha=0.8)

        plt.ylabel(y_label, fontproperties=self.font(15))
        plt.yticks(fontproperties=self.font(15))
        plt.xlabel(x_label, fontproperties=self.font(15))
        plt.xticks(fontproperties=self.font(15))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")

        ax2 = fig.add_subplot(2, 1, 2)
        error = data[np.abs(data - mean) > 3 * std]
        data_c = data[np.abs(data - mean) <= 3 * std]
        plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
        plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)

        plt.ylabel(x_label, fontproperties=self.font(15))
        plt.yticks(fontproperties=self.font(15))
        plt.xlabel("样本编号", fontproperties=self.font(15))
        plt.xticks(fontproperties=self.font(15))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")

        plt.savefig("./image/3delta_{}.jpg".format(x_label.strip().replace("/", "")))
        plt.close()



    @CyrusPlot()
    def boxplot(self,data,x_label=[],y_label=[],type="single",path="",whis = 3):
        """
        :param data: numpy or pandas
        :param col:
        :param type: single or multi
        :param whis:
        :return:
        """
        data = np.array(data)
        if not x_label:
            y_label = []
            if len(data.shape) == 1 or data.shape[1] == 1:
                x_label = ["x"]
                y_label = ["y"]
                data = data.reshape([-1,1])
            else:
                for i in range(1,data.shape[1]+1):
                   x_label.append("x{}".format(i))
                   y_label.append("y{}".format(i))
        y_label = y_label if y_label else [" "] * len(x_label)

        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        if type == "single":
            for i in range(len(x_label)):
                self.logger.info("{} is plotting boxplot!".format(x_label[i]))
                figure = plt.figure(figsize=[8, 12], dpi=72)
                sns.boxplot(data=data[:,i], whis=whis, orient="v", fliersize=15)
                plt.ylabel(y_label[i], fontproperties=self.font(30))
                plt.yticks(fontproperties=self.font(30))
                plt.xlabel(x_label[i], fontproperties=self.font(30))
                plt.xticks(fontproperties=self.font(30))
                plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
                plt.savefig(("./image/box_"+x_label[i].replace("/","_")+"_"+path + ".png").replace(" ",""))
                plt.close()
        else:
            row = np.ceil(len(x_label)/3)
            figure = plt.figure(figsize=[30, 12*row], dpi=72)
            for i in range(len(x_label)):
                self.logger.info("{} is plotting!".format(x_label[i]))
                figure.add_subplot(row, 3, i + 1)
                sns.boxplot(data=data[:,i], whis=whis, orient="v", fliersize=15)
                plt.ylabel(y_label[i], fontproperties=self.font(30))
                plt.yticks(fontproperties=self.font(30))
                plt.xlabel(x_label[i], fontproperties=self.font(30))
                plt.xticks(fontproperties=self.font(30))
                plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
            plt.savefig(("./image/box_"+path + ".png").replace(" ",""))
            plt.close()



    @CyrusPlot()
    def distplot(self,data,x_label=[],type="single",path=""):
        """

        :param data:
        :param x_label:
        :param type: single or multi
        :param path:
        :return:
        """
        data = np.array(data)
        if not x_label:
            if len(data.shape) == 1 or data.shape[1] == 1:
                x_label = ["x"]

                data = data.reshape([-1, 1])
            else:
                for i in range(1, data.shape[1] + 1):
                    x_label.append("x{}".format(i))

        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        if type == "single":
            for i in range(len(x_label)):
                self.logger.info("{} is plotting distplot!".format(x_label[i]))
                figure = plt.figure(figsize=[12, 8], dpi=72)
                sns.distplot(data[:, i], bins=20, kde_kws={"color": "red",'bw' : 1}, color="c")
                plt.ylabel("概率密度", fontproperties=self.font(30))
                plt.yticks(fontproperties=self.font(30))
                plt.xlabel(x_label[i], fontproperties=self.font(30))
                plt.xticks(fontproperties=self.font(30))
                plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
                plt.savefig(("./image/dist_" + x_label[i].replace("/","_") + "_" + path + ".png").replace(" ",""))
                plt.close()
        else:
            row = np.ceil(len(x_label) / 3)
            figure = plt.figure(figsize=[36, 20 * row], dpi=72)
            for i in range(len(x_label)):
                self.logger.info("{} is plotting!".format(x_label[i]))
                figure.add_subplot(row, 3, i + 1)
                sns.distplot(data[:, i], bins=20, kde_kws={"color": "red"}, color="c")
                plt.ylabel("概率密度", fontproperties=self.font(30))
                plt.yticks(fontproperties=self.font(30))
                plt.xlabel(x_label[i], fontproperties=self.font(30))
                plt.xticks(fontproperties=self.font(30))
                plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
            plt.savefig("./image/dist_" + path + ".png")
            plt.close()

    @CyrusPlot()
    def heatmap(self,corrs,x_ticks = [],y_ticks = [],path ="",bar_label = "r"):

        """
        :param corrs: numpy
        :param x_ticks:
        :param y_ticks:
        :param bar_label:
        :param show:
        :param save_name:
        :return:
        """
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        figure = plt.figure(figsize=[30, 20], dpi=72)
        ax = figure.add_subplot(111)
        if not x_ticks:
            x_ticks = ["x" + str(i) for i in range(corrs.shape[1])]
            y_ticks = ["y" + str(i) for i in range(corrs.shape[0])]
        im, _ = self.sub_heatmap_1(np.array(corrs), x_ticks, y_ticks, cmap="RdBu", cbarlabel=bar_label, ax=ax)  # plt.cm.RdBu   PuOr
        self.annotate_heatmap(im, valfmt="{x:.2f}", size=12)
        plt.savefig("./image/corr_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def cluster_scatter_2d(self,x,y,classes,x_label="x",y_label="y",path = None):
        """
        :param x:
        :param y:
        :param classes: 聚类的类别
        :param x_label:
        :param y_label:
        :param path:
        :return:
        """
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        x = np.array(x).reshape([-1,1])
        y = np.array(y).reshape([-1, 1])
        data = np.array(np.concatenate([x,y],axis=-1))
        data = pd.DataFrame(data,columns=["x_{}".format(i) for i in range(1,data.shape[1]+1)])
        data["label"] = classes
        center = data.groupby("label").mean()
        idx = center.index.to_list()
        figure = plt.figure(figsize=[30, 16], dpi=72)
        for count,i in enumerate(idx):
            tmp = data[data["label"] == i]
            x,y = tmp.iloc[:,0],tmp.iloc[:,1]
            plt.scatter(x,y,c=self.color[count],marker=self.marker[count],s=150)
            plt.scatter([center.loc[i,:][0]], [center.loc[i,:][1]], c=self.color[count], marker=self.marker[count], s=1000,label="第{}类聚类中心".format(count+1))
        x,y = data.iloc[:,0],data.iloc[:,1]
        x_min, x_max = np.array(x).min(), np.array(x).max()
        y_min, y_max = np.array(y).min(), np.array(y).max()
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        x_min, x_max = x_min - 0.05 * delta_x, x_max + 0.05 * delta_x
        y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.xlabel(x_label, fontproperties=self.font(45))
        plt.ylabel(y_label, fontproperties=self.font(45))
        plt.legend(prop =self.font(45))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/cluster_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def cluster_scatter_3d(self, x, y, z,classes, x_label="x", y_label="y", z_label="z",path=None,show=False):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection="3d")
        x = np.array(x).reshape([-1, 1])
        y = np.array(y).reshape([-1, 1])
        z = np.array(z).reshape([-1, 1])
        data = np.array(np.concatenate([x, y,z], axis=-1))
        data = pd.DataFrame(data, columns=["x_{}".format(i) for i in range(1, data.shape[1] + 1)])
        data["label"] = classes
        center = data.groupby("label").mean()
        idx = center.index.to_list()
        for count, i in enumerate(idx):
            tmp = data[data["label"] == i]
            x, y, z = tmp.iloc[:, 0], tmp.iloc[:, 1], tmp.iloc[:, 2]
            ax1.scatter(x, y, z, zdir="z", c=self.color[count], marker=self.marker[count], s=20)
            ax1.scatter([center.loc[i, :][0]], [center.loc[i, :][1]], [center.loc[i, :][2]], zdir="z", c=self.color[count],
                        marker=self.marker[count], s=150,
                        label="第{}类聚类中心".format(count + 1))

        ax1.set_xlabel(x_label, fontproperties=self.font(10))
        ax1.set_ylabel(y_label, fontproperties=self.font(10))
        ax1.set_zlabel(z_label, fontproperties=self.font(10))
        plt.legend(prop=self.font(10))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/cluster_3d_" + path + ".png")
        if show:
            plt.show()
        plt.close()

    @CyrusPlot()
    def dbscan_scatter_2d(self,data,labels,path=None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        idxs = set(labels)
        labels = pd.DataFrame(np.stack([np.arange(data.shape[0]),labels],axis=1),columns=["idx","label"])

        figure = plt.figure(figsize=[20, 20], dpi=72)
        color = ["g",  "b", "y", "c", "m","r",]
        marker = ["o",  "s", "+", "x", "v","*",]
        cnt = 0
        for idx in idxs:
            tmp = data[labels[labels["label"]==idx]["idx"].to_list()]
            x = tmp[:,0]
            y = tmp[:,1]
            if idx == -1:
                plt.scatter(x, y, c=color[-1], marker=marker[-1], s=150,label="noise")
                continue
            plt.scatter(x, y, c=color[cnt], marker=marker[cnt], s=150,label="class_{}".format(cnt + 1))
            cnt += 1
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/dbscan_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def actual_pre_line(self,y,y_pre,path = "", x_label="样本编号",y_label="y",legend = ["实际值","预测值"]):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        figure = plt.figure(figsize=[20,12], dpi=72)
        x = np.arange(np.array(y).reshape([-1, ]).shape[0])
        plt.plot(x, y, color="c", label=legend[0], marker='o', markersize=6, markeredgecolor="b",
                 markerfacecolor="b")
        plt.scatter(x, y_pre, s=60, label=legend[1], color="r", marker="*")
        plt.legend(prop=self.font(30), loc="best")
        x_min, x_max = np.array(x).min(), np.array(x).max()
        y_min, y_max = np.array(y).min(), np.array(y).max()
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        x_min, x_max = x_min - 0.1 * delta_x, x_max + 0.1 * delta_x
        y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.xlabel(x_label, fontproperties=self.font(30))
        plt.ylabel(y_label, fontproperties=self.font(30))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/pre_model_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def lossplot(self,train_loss,val_loss=None,path = "", x_label="迭代次数",y_label="损失函数值",legend = ["训练集","验证集"]):
        train_loss = np.array(train_loss)
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        figure = plt.figure(figsize=[20,12], dpi=72)
        step = np.arange(1,train_loss.shape[0]+1)
        plt.plot(step, train_loss, color="r", label=legend[0])
        if val_loss:
            val_loss = np.array(val_loss)
            plt.plot(step, val_loss, color="c",label = legend[1])
            plt.legend(prop = self.font(30))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.xlabel(x_label, fontproperties=self.font(30))
        plt.ylabel(y_label, fontproperties=self.font(30))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/loss_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def r2plot(self,y,y_pre,path = ""):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        figure = plt.figure(figsize=[20,12], dpi=72)
        x_min, x_max = np.array(y_pre).min(), np.array(y_pre).max()
        y_min, y_max = np.array(y).min(), np.array(y).max()
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        x_min, x_max = x_min - 0.1 * delta_x, x_max + 0.1 * delta_x
        y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        val_min = min(x_min,y_min)
        val_max = max(x_max,y_max)

        plt.plot([val_min,val_max], [val_min,val_max], color="black")
        plt.scatter(y_pre,y, s=60, color="r", marker="*")
        plt.xlabel("预测值", fontproperties=self.font(30))
        plt.ylabel("实际值", fontproperties=self.font(30))
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/r2_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def silhouette_plot(self,silhouette_sample,labels,path=None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        if type(silhouette_sample) == pd.DataFrame:
            silhouette_sample = silhouette_sample.to_numpy()[:,0]
        idxs = np.unique(labels)
        idxs.sort()
        low,up = 0,0
        yticks = []
        figure = plt.figure(figsize=[20, 12], dpi=72)
        for i,idx in enumerate(np.unique(labels)):
            silhouette = silhouette_sample[labels==idx]
            silhouette.sort()
            up += silhouette.shape[0]
            plt.barh(np.arange(low,up),silhouette,height=1,edgecolor=None,color=self.color[i])
            yticks.append((low+up)/2)
            low += silhouette.shape[0]
        # silhouette_mean = silhouette_sample.mean()
        # plt.axvline(silhouette_mean,color="read",linestyle="--")
        plt.yticks(yticks,["第{}簇".format(i) for i in range(1,idxs.shape[0]+1)],fontproperties=self.font(30))
        plt.ylabel("样本",fontproperties=self.font(30))
        plt.xlabel("轮廓系数",fontproperties=self.font(30))
        plt.savefig("./image/silhouette_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def tree_image(self,data,dist_method = "average",path=None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        fig, ax = plt.subplots(figsize=(8, 8))
        data = linkage(data, dist_method)
        dendrogram(data, leaf_font_size=14)
        plt.title("系统聚类 {}".format(dist_method),fontproperties=self.font(30))
        plt.xlabel("样本编号",fontproperties=self.font(30))
        plt.ylabel("距离",fontproperties=self.font(30))
        plt.savefig("./image/tree_image_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def line_scatter(self,x,y,x_label="x",y_label="y",path=None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")

        figure = plt.figure(figsize=[30, 16], dpi=72)
        plt.plot(x, y, color="b", marker="o", markersize=10, markeredgecolor='r', markeredgewidth=10)
        x_min, x_max = np.array(x).min(), np.array(x).max()
        y_min, y_max = np.array(y).min(), np.array(y).max()
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        x_min, x_max = x_min - 0.05 * delta_x, x_max + 0.05 * delta_x
        y_min, y_max = y_min - 0.1 * delta_y, y_max + 0.1 * delta_y
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xticks(fontproperties=self.font(30))
        plt.yticks(fontproperties=self.font(30))
        plt.xlabel(x_label, fontproperties=self.font(45))
        plt.ylabel(y_label, fontproperties=self.font(45))
        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/line_scatter_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def grid_3d_plot(self,x,y,z,x_label="x",y_label = "y",z_label = "z",path=None):
        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 1, 1, projection="3d")
        X, Y = np.meshgrid(x, y)
        Z = z
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax1.set_xlabel(x_label, fontdict={"size":30})
        ax1.set_ylabel(y_label, fontdict={"size":30})
        ax1.set_zlabel(z_label, fontdict={"size":30})
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("./image/grid_3d_plot_" + path + ".png")
        plt.close()

    @CyrusPlot()
    def sub_heatmap_1(self,data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontproperties=FontProperties(fname="C:\Windows\Fonts\simhei.ttf", size=30))

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels,fontproperties=FontProperties(fname="C:\Windows\Fonts\simhei.ttf", size=30))
        ax.set_yticklabels(row_labels,fontproperties=FontProperties(fname="C:\Windows\Fonts\simhei.ttf", size=30))

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                    rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im,cbar

    @CyrusPlot()
    def annotate_heatmap(self,im, data=None, valfmt="{x:.2f}",
                            textcolors=("black", "white"),
                            threshold=None, **textkw):

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                    verticalalignment="center",
                    )
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[abs(data[i, j]) > 0.5])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    @CyrusPlot()
    def multi_hist(self,data,bins = 15,x_label="x",y_label = "y",path=None):
        assert type(data) == list or type(data) == tuple, "请传入列表或者元组对象"

        if not path:
            path = datetime.datetime.now().strftime("%Y-%m-%d")

        x_min, x_max = min([min(i) for i in data]),max([max(i) for i in data])

        n = len(data)
        fig = plt.figure(figsize=(16, 4*n))
        for cnt,x in enumerate(data):
            fig.add_subplot(n,1,cnt+1)
            plt.hist(x,bins=bins,color="c")
            plt.xlim((x_min, x_max))
            plt.yticks(fontproperties=self.font(30))
            if cnt + 1 == n:
                plt.xlabel(x_label, fontproperties=self.font(45))
                plt.xticks(fontproperties=self.font(30))
            else:
                plt.xticks([],roperties=self.font(30))
            plt.ylabel(y_label, fontproperties=self.font(45))

        plt.grid(linestyle="--", linewidth=1, alpha=0.5, axis="y")
        plt.savefig("./image/multi_hist" + path + ".png")
        plt.close()

if __name__ == '__main__':
    plot_tool = PlotTool()
    import scipy.stats as ss
    # plot_tool.boxplot(np.zeros([100,2]),path="")
    # plot_tool.multi_hist([np.random.randn(15),np.random.randn(25),np.random.randn(19),])
    plot_tool.QQPlot(ss.norm.rvs(size=100))
