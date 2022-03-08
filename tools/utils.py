# -*- coding: utf-8 -*-
# @Time : 2022/1/5 12:48
# @Author : CyrusMay WJ
# @FileName: utils.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May
import os
import pandas as pd
from sklearn.manifold import TSNE
import logging
import sys
from sklearn.model_selection import train_test_split as split
import pickle as pkl
import datetime

logger = logging.getLogger(name="save_excel") #
logger.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(sys.stdout)
screen_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

def save_to_excel(data, path):
    if "file" not in os.listdir():
        os.mkdir("./file")
    if type(data) == list:
        with pd.ExcelWriter("./file/" + path + ".xlsx") as f:
            for df,sheet_name in data:
                df.to_excel(f, sheet_name=sheet_name)
        logger.info("./file/" + path + ".xlsx" + " has been saved!")
        return
    assert type(data) == pd.DataFrame,"The type of data is not DataFrame!"
    with pd.ExcelWriter("./file/" + path + ".xlsx") as f:
        data.to_excel(f)

    logger.info(os.path.abspath(os.path.dirname(__file__)) + "\\file\\" + path + ".xlsx" + " has been saved!")


def decomp_by_tsne(data,n_dims=2):
    tsne = TSNE(n_components=n_dims)
    result = tsne.fit_transform(data)
    return result


def save_var(var,path):
    if "vars" not in os.listdir():
        os.mkdir("./vars")
    with open("./vars/" +path + ".pkl","wb") as f:
        pkl.dump(var,f)

def train_test_split(x,y,test_size=0.2):
    x_train,x_test,y_train,y_test = split(x,y,test_size=test_size)
    return x_train,x_test,y_train,y_test

def str_to_date(string,fmt="%Y-%m-%d %H:%M:%S"):
    if type(string) == datetime.datetime:
        return string
    return datetime.datetime.strptime(string,fmt)

def date_to_str(date,fmt="%Y-%m-%d %H:%M:%S"):
    return date.strftime(fmt)

