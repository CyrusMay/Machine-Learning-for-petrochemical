# -*- coding: utf-8 -*-
# @Time : 2022/1/12 23:05
# @Author : CyrusMay WJ
# @FileName: OptTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May
import tensorflow as tf
import functools
import datetime
import pickle as pkl
import joblib

from sko.GA import GA
from sko.operators import ranking, selection, crossover, mutation
from tools.PlotTool import PlotTool
import os
import numpy as np
from sko.PSO import PSO
from sko.SA import SA
from sko.DE import DE
from tools.utils import save_to_excel
import pandas as pd


class CyrusOptMessage():
    def __init__(self,message=""):
        self.message = message

    def __call__(self, func):
        @functools.wraps(func)
        def _deco(*args,**kwargs):
            print("*" * 20 + "start to build optimizer for {} in {}!".format(self.message,
                                                                         datetime.datetime.now().strftime(
                                                                     "%Y-%m-%d")) + "*" * 20)
            func(*args,**kwargs)
        return _deco

class CyrusOptimizer():
    def __init__(self,model_path = None,standard_tool_path=None,is_min=True,var_is_opt = [True],lb=[],up=[],logger=None):
        self.logger = logger
        self.is_min = is_min
        self.plot_tool = PlotTool(logger)
        self.is_opt_idx = [i for i in range(len(var_is_opt)) if var_is_opt[i]]
        self.not_opt_idx = [i for i in range(len(var_is_opt)) if not var_is_opt[i]]
        self.__load_model(model_path)
        self.__load_standard_tool(standard_tool_path)
        self.__load_range(lb,up)

    def build_opt(self):
        self.opt_name = None
        self.opt_tool = None
    def _build(self):
        self.opt_name = None
        self.opt_tool = None

    def __load_range(self,lb,up):
        self.lb = list(np.array(lb)[self.is_opt_idx])
        self.up = list(np.array(up)[self.is_opt_idx])

    def __load_standard_tool(self,standard_tool_path):
        with open("{}".format(standard_tool_path),"rb") as f:
            self.standard_tool = pkl.load(f)

    def __load_model(self,model_path):
        if ".m" in model_path:
            self.model = joblib.load("./model/{}".format(model_path))
        else:
            self.model = tf.keras.models.load_model("{}".format(model_path))

    def get_obj(self):
        def obj(x):
            input_x = np.zeros(len(self.is_opt_idx) + len(self.not_opt_idx))
            input_x[self.not_opt_idx] = self.preopt_var[self.not_opt_idx]
            for i,j in enumerate(self.is_opt_idx):
                input_x[j] = x[i]
            x_std = self.standard_tool.transform_x(input_x[np.newaxis,:])
            y_std = self.model.predict(x_std)
            try:
                y = self.standard_tool.inverse_y(y_std)[0][0]
            except:
                y = self.standard_tool.inverse_y(y_std)[0]
            if self.is_min:
                return y
            else:
                return -y
        return obj

    def predict(self,x):
        x = np.array(x)
        x_std = self.standard_tool.transform_x(x[np.newaxis, :])
        y_std = self.model.predict(x_std)
        try:
            y = self.standard_tool.inverse_y(y_std)[0][0]
        except:
            y = self.standard_tool.inverse_y(y_std)[0]
        if self.is_min:
            return -y
        else:
            return y

    def run(self,preopt_var,is_plot=False):
        self.preopt_var = preopt_var
        op_str = "self._build(" + ",".join(["self.params[{}]".format(i) for i in range(len(self.params))]) + ")"
        eval(op_str)
        best_x, best_y = self.opt_tool.run()
        best_y = best_y if self.is_min else -best_y
        x = preopt_var
        x[self.is_opt_idx] = best_x
        try:
            y_history = (-np.array(self.opt_tool.all_history_Y)).max(axis=1) if not self.is_min else (np.array(self.opt_tool.all_history_Y)).max(axis=1)
            if is_plot:
                self.plot_tool.lossplot(y_history,x_label="优化次数",y_label="目标函数值",path="{}_opt_iter".format(self.opt_name))
                save_to_excel(pd.DataFrame(y_history[:,None],columns=["目标函数值"]),path="{}_opt_iter".format(self.opt_name))
        except:
            pass
        best_y = best_y[0] if type(best_y)!=np.float32 else best_y
        return x,best_y

    def run_by_step(self, preopt_var, is_plot=False):
        self.preopt_var = preopt_var
        op_str = "self._build(" + ",".join(["self.params[{}]".format(i) for i in range(len(self.params))]) + ")"
        eval(op_str)
        history = []
        for step in range(200):
            best_x, best_y = self.opt_tool.run(1)
            history.append(-best_y)
        history = np.array(history)

        try:

            if is_plot:
                self.plot_tool.lossplot(history, x_label="优化次数", y_label="目标函数值",
                                        path="{}_opt_iter".format(self.opt_name))
                save_to_excel(pd.DataFrame(history[:,0,0][:, None], columns=["目标函数值"]),
                              path="{}_opt_iter".format(self.opt_name))
        except:
            pass
        best_y = best_y[0] if type(best_y) != np.float32 else best_y
        return best_y

class GAOptimizer(CyrusOptimizer):
    def __init__(self,*args,**kwargs):
        super(GAOptimizer,self).__init__(*args,**kwargs)

    @CyrusOptMessage("GA")
    def build_opt(self,size_pop=50,max_iter=200,prob_mut=0.01,):
        self.opt_name = "GA"
        self.params = [size_pop,max_iter,prob_mut]

    def _build(self,size_pop=50,max_iter=200,prob_mut=0.01):
        params = {
            "func": self.get_obj(),
            "n_dim": len(self.is_opt_idx),
            "size_pop": size_pop,
            "max_iter": max_iter,
            "prob_mut": prob_mut,
            "lb": self.lb,
            "ub": self.up,
            "precision": 1e-7
        }
        self.opt_tool = GA(**params)
        self.opt_tool.register(operator_name="selection", operator=selection.selection_roulette_1)
        self.opt_tool.register(operator_name="crossover", operator=crossover.crossover_2point_bit)
        self.opt_tool.register(operator_name="mutation", operator=mutation.mutation)

class PSOOptimizer(CyrusOptimizer):
    def __init__(self,*args,**kwargs):
        super(PSOOptimizer,self).__init__(*args,**kwargs)

    @CyrusOptMessage("PSO")
    def build_opt(self,pop=50,max_iter=200,w=0.8,c1=0.5,c2=0.5):
        self.opt_name = "PSO"
        self.params = [pop,max_iter,w,c1,c2]

    def _build(self,pop=50,max_iter=200,w=0.8,c1=0.5,c2=0.5):
        params = {
            "func": self.get_obj(),
            "dim": len(self.is_opt_idx),
            "pop": pop,
            "max_iter": max_iter,
            "lb": self.lb,
            "ub": self.up,
            "w": w,
            "c1": c1,
            "c2": c2
        }
        self.opt_tool = PSO(**params)


class SAOptimizer(CyrusOptimizer):
    def __init__(self,*args,**kwargs):
        super(SAOptimizer,self).__init__(*args,**kwargs)

    @CyrusOptMessage("SA")
    def build_opt(self,T_max=100,T_min=1e-7,L=300,max_stay_counter=200):
        self.opt_name = "SA"
        self.params = [T_max,T_min,L,max_stay_counter]

    def _build(self,T_max=100,T_min=1e-7,L=300,max_stay_counter=200):
        params = {
            "func": self.get_obj(),
            "x0": list(self.preopt_var[self.is_opt_idx]),
            "T_max": T_max,
            "T_min": T_min,
            "L": L,
            "max_stay_counter": max_stay_counter,
            "lower": np.array(self.lb),
            "upper": np.array(self.up)
        }
        self.opt_tool = SA(**params)

class DEOptimizer(CyrusOptimizer):
    def __init__(self,*args,**kwargs):
        super(DEOptimizer,self).__init__(*args,**kwargs)

    @CyrusOptMessage("DE")
    def build_opt(self,size_pop=50,max_iter=800):
        self.opt_name = "DE"
        self.params = [size_pop,max_iter]

    def _build(self,size_pop=50,max_iter=800):
        params = {
            "func": self.get_obj(),
            "n_dim": len(self.is_opt_idx),
            "size_pop": size_pop,
            "max_iter": max_iter,
            "lb" :self.lb,
            "ub": self.up,

        }
        self.opt_tool = DE(**params)



