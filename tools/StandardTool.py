# -*- coding: utf-8 -*-
# @Time : 2022/1/5 13:35
# @Author : CyrusMay WJ
# @FileName: StandardTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May

from sklearn.preprocessing import MinMaxScaler,StandardScaler
class StandardTool():
    def __init__(self,x,y=None,y_dims = None,logger=None,type = "standard",):
        """
        :param x:
        :param y:
        :param y_dims: 0 or 1 or 2
        :param logger:
        :param type: standard or minmax
        """
        self.logger = logger
        self.x = x
        self.obj_x = StandardScaler() if type == "standard" else MinMaxScaler()
        self.y = y
        self.y_dims = y_dims
        if self.y_dims:
            if self.y_dims == 1:
                if len(self.y.shape) == 1:
                    self.y = self.y[:,None]

            self.obj_y = StandardScaler() if type == "standard" else MinMaxScaler()

        self.__fit()

    def __fit(self):
        self.obj_x.fit(self.x)
        if self.y_dims:
            self.obj_y.fit( self.y)
    def transform_x(self,x):
        return self.obj_x.transform(x)

    def transform_y(self,y):
        if self.y_dims == 1 and len(y.shape) == 1:
                y = y[:, None]
        return self.obj_y.transform(y)[:,0] if self.y_dims == 1 else self.obj_y.transform(y)

    def inverse_x(self,x):
        return self.obj_x.inverse_transform(x)

    def inverse_y(self,y):
        if self.y_dims == 1 and len(y.shape) == 1:
            y = y[:, None]
        return self.obj_y.inverse_transform(y)[:, 0] if self.y_dims == 1 else self.obj_y.inverse_transform(y)





