# -*- coding: utf-8 -*-
# @Time : 2022/1/14 20:03
# @Author : CyrusMay WJ
# @FileName: FeatureTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May

import os
import pandas as pd
from tools.PlotTool import PlotTool
from tools.utils import save_to_excel,decomp_by_tsne
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from tools.MetricsTool import CyrusMetrics

from tools.StandardTool import StandardTool
import functools
import datetime
from tools.utils import save_var
from factor_analyzer import FactorAnalyzer
from minepy import MINE
from scipy.stats import pearsonr


class CyrusPreprocessMessage():
    def __init__(self,message=None):
        self.message = message

    def __call__(self, func):
        @functools.wraps(func)
        def _deco(*args,**kwargs):

            print("*" * 20 + "start to use feature_tool ({}) in {}!".format(self.message,
                datetime.datetime.now().strftime("%Y-%m-%d")) + "*" * 20)
            func(*args, **kwargs)
        return _deco

class CorrelationsTool():
    def __init__(self, logger=None):
        self.logger = logger
        self.plot_tool = PlotTool(logger=logger)
    @CyrusPreprocessMessage("pearson")
    def pearson(self, X):
        assert type(X) == pd.DataFrame, "please input DataFrame object!"
        cols = X.columns.to_list()
        standard_tool = StandardTool(x=X.to_numpy(), logger=self.logger)
        x_std = standard_tool.transform_x(X.to_numpy())
        corrs = pd.DataFrame(x_std, columns=cols).corr()
        self.plot_tool.heatmap(corrs, x_ticks=cols, y_ticks=cols, path="pearson")
        p_val = np.ones_like(corrs.to_numpy())
        for i in range(x_std.shape[1]):
            for j in range(x_std.shape[1]):
                p_val[i,j] = pearsonr(x=x_std[:,i],y=x_std[:,j])[1]
        p_val = pd.DataFrame(p_val,columns=cols,index=cols)
        save_to_excel([(corrs,"pearson"),(p_val,"p_val")] ,path="pearson")

    @CyrusPreprocessMessage("spearman")
    def spearman(self, X):
        assert type(X) == pd.DataFrame, "please input DataFrame object!"
        cols = X.columns.to_list()
        standard_tool = StandardTool(x=X.to_numpy(), logger=self.logger)
        x_std = standard_tool.transform_x(X.to_numpy())
        corrs = pd.DataFrame(x_std, columns=cols).corr(method="spearman")
        self.plot_tool.heatmap(corrs, x_ticks=cols, y_ticks=cols, path="spearman")
        save_to_excel(corrs, path="spearman")

    @CyrusPreprocessMessage("mutal_information")
    def mutal_information(self,x,y,columns = []):
        if not columns:
            if type(x) == pd.DataFrame:
                columns = x.columns.to_list()
            else:
                columns = ["x{}".format(i) for i in range(x.shape[1])]
        standard_tool = StandardTool(x=x,y=y,y_dims=1, logger=self.logger)
        x_std = standard_tool.transform_x(x)
        y_std = standard_tool.transform_y(y)
        mis = mutual_info_regression(x_std,y_std)

        save_to_excel(pd.DataFrame(mis[:,None],index=columns,columns=["mutal_information"]),path="mutal_information")
        mis.sort()
        self.plot_tool.line_scatter(x= [i for i in range(x.shape[1])],y=mis[::-1],x_label="feature",y_label="mutal_information",path="mutal_information")

    @CyrusPreprocessMessage("MIC")
    def mic(self,x,y,columns = []):
        if not columns:
            if type(x) == pd.DataFrame:
                columns = x.columns.to_list()
            else:
                columns = ["x{}".format(i) for i in range(x.shape[1])]
        standard_tool = StandardTool(x=x, y=y, y_dims=1, logger=self.logger)
        x_std = standard_tool.transform_x(x)
        y_std = standard_tool.transform_y(y)

        mine_tool = MINE(alpha=0.6, c=15)
        mics = []
        for i in range(x_std.shape[1]):
            mine_tool.compute_score(x_std[:, i], y_std)
            mics.append(mine_tool.mic())
        mics = np.array(mics)
        save_to_excel(pd.DataFrame(mics[:, None], index=columns, columns=["MIC"]),
                      path="MIC")
        mics.sort()
        self.plot_tool.line_scatter(x=[i for i in range(x.shape[1])], y=mics[::-1], x_label="feature",
                                    y_label="MIC", path="MIC")

    @CyrusPreprocessMessage("tree_feature_importance_xgb")
    def tree_feature_importance(self,x,y,columns = []):
        if not columns:
            if type(x) == pd.DataFrame:
                columns = x.columns.to_list()
            else:
                columns = ["x{}".format(i) for i in range(x.shape[1])]
        standard_tool = StandardTool(x=x, y=y, y_dims=1, logger=self.logger)
        x_std = standard_tool.transform_x(x)
        y_std = standard_tool.transform_y(y)

        model = XGBRegressor()
        model.fit(x_std,y_std)
        feature_importance = model.feature_importances_
        save_to_excel(pd.DataFrame(feature_importance[:, None], index=columns, columns=["feature_importance"]),
                      path="feature_importance_xgb")
        feature_importance.sort()
        self.plot_tool.line_scatter(x=[i for i in range(x.shape[1])], y=feature_importance[::-1], x_label="feature",
                                    y_label="feature_importance_xgb", path="feature_importance_xgb")



class CyrusFactorAnalysis():
    def __init__(self,logger=None):
        self.logger = logger
        self.metric_tool = CyrusMetrics(logger=self.logger)
        self.plot_tool = PlotTool(self.logger)

    def select_factor_nums(self,data):
        self.standard_tool = StandardTool(data)
        std_data = self.standard_tool.transform_x(data)
        self.factor_tool = FactorAnalyzer(n_factors=data.shape[1], rotation="promax")
        var = self.factor_tool.get_factor_variance()
        save_to_excel()

    def run_factor_analysis(self,data,n_factor=2):
        self.standard_tool = StandardTool(data)
        std_data = self.standard_tool.transform_x(data)
        self.factor_tool = FactorAnalyzer(n_factors=n_factor, rotation="promax")

        process_data = self.factor_tool.fit_transform(std_data)
        factor_data = self.factor_tool.loadings_
        weights = self.factor_tool.weights_
        var = self.factor_tool.get_factor_variance()

        save_to_excel([(pd.DataFrame(factor_data),"载荷矩阵"),(pd.DataFrame(process_data),"归因后结果"),
                       (pd.DataFrame(weights),"归因系数"),(pd.DataFrame(var),"方差解释性")],
                      path="FactorAnalysisResult_{}".format(datetime.datetime.now().strftime("%Y-%m-%d")))

    def transform(self,data):
        std_data = self.standard_tool.transform_x(data)
        factor_data = self.factor_tool.transform(std_data)
        return factor_data

    def save_model(self):
        save_var(self.factor_tool,path="FactorAnalysisModel_{}".format(datetime.datetime.now().strftime("%Y-%m-%d")))


class DataAnalysis():
    def __init__(self,logger=None):
        self.logger = logger

    def cal_distribution(self,data,columns = [],path="数据分布"):
        if not columns:
            if type(data) == pd.DataFrame:
                columns = data.columns.to_list()
                data = data.to_numpy()

            else:
                columns = ["x{}".format(i) for i in range(data.shape[1])]
        data = pd.DataFrame(data,columns=columns)
        index = ["上四分位数","下四分位数","中位数","平均值","最小值","最大值","标准差","方差","偏态系数","峰态系数"]
        result = []
        result.append(data.quantile(q=0.75).to_numpy())
        result.append(data.quantile(q=0.25).to_numpy())
        result.append(data.quantile(q=0.5).to_numpy())
        result.append(data.mean().to_numpy())
        result.append(data.min().to_numpy())
        result.append(data.max().to_numpy())
        result.append(data.std().to_numpy())
        result.append(data.var().to_numpy())
        result.append(data.skew().to_numpy())
        result.append(data.kurt().to_numpy())
        save_to_excel(pd.DataFrame(result,index=index,columns=columns),path=path)

