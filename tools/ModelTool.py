# -*- coding: utf-8 -*-
# @Time : 2022/1/5 19:43
# @Author : CyrusMay WJ
# @FileName: ModelTool.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May

import functools
import datetime
from tools.StandardTool import StandardTool
from sklearn.svm import SVR
from tools.MetricsTool import CyrusMetrics
from tools.PlotTool import PlotTool
from tools.utils import save_var,save_to_excel
from xgboost import XGBRegressor
import tensorflow as tf
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow.keras.backend as K

import pandas as pd
from sklearn.model_selection import GridSearchCV
import pickle as pkl
import time
import shutil

import webbrowser
import subprocess





class CyrusModelMessage():
    def __init__(self,model_name=None):
        self.model_name = model_name

    def __call__(self,func):
        @functools.wraps(func)
        def _deco(*args,**kwargs):
            print("*"*20 + "start to build model for {} in {}!".format(self.model_name,datetime.datetime.now().strftime("%Y-%m-%d"))+"*"*20)
            return func(*args,**kwargs)
        return _deco

class CyrusModel():
    def __init__(self,x,y,logger=None):
        if "logs" not in os.listdir():
            os.mkdir("./logs")
        self.logger = logger
        self.metric_tool = CyrusMetrics(logger=self.logger)
        self.plot_tool = PlotTool(self.logger)
        self.x = x
        self.y = y
        self.__standard_data()

    @CyrusModelMessage()
    def build_model(self):
        self.model_name = None
        self.model = None



    def _load_tensorboard(self):
        self.logger.info("start to load tensorboard, please wait!")
        if os.path.exists("./logs/log_loss"):
            shutil.rmtree("./logs/log_loss")
        subprocess.Popen("tensorboard --logdir ./logs/log_loss",shell=False)
        time.sleep(5)
        webbrowser.open("http://localhost:6006")
        self.logger.info("load tensorboard successfully,please click http://localhost:6006 if not jump!")


    def __standard_data(self):
        if len(self.y.shape) > 1:
            if self.y.shape[1] == 1:
                self.y = self.y[:,0]
                self.standard_tool =StandardTool(self.x,self.y,y_dims=1)
            else:
                self.standard_tool = StandardTool(self.x, self.y, y_dims=2)
        else:
            self.standard_tool = StandardTool(self.x, self.y, y_dims=1)
        self.x_std = self.standard_tool.transform_x(self.x)
        self.y_std = self.standard_tool.transform_y(self.y)
        if "pkl_file" not in os.listdir():
            os.mkdir("./pkl_file")
        with open("./pkl_file/standard_tool.pkl".format(),"wb") as f:
            pkl.dump(self.standard_tool,f)


    def fit(self):
        self.model.fit(self.x_std,self.y_std)
        self.save_model()

    def predict(self,x):
        return self.standard_tool.inverse_y(self.model.predict(self.standard_tool.transform_x(x)))

    def evaluate(self,x_test,y_test):
        y_pre = self.standard_tool.inverse_y(self.model.predict(self.x_std))
        y_test_pre = self.standard_tool.inverse_y(self.model.predict(self.standard_tool.transform_x(x_test)))
        metric_train = self.metric_tool.regression_metrics(self.y,y_pre,path=self.model_name + "_train")
        metric_test = self.metric_tool.regression_metrics(y_test,y_test_pre,path=self.model_name + "_test")


        self.logger.info("metrics for train:")
        self.logger.info(metric_train)
        self.logger.info("metrics for test:")
        self.logger.info(metric_test)

    def plot_pre(self,x_test,y_test,loss=None,val_loss=None):
        y_pre = self.standard_tool.inverse_y(self.model.predict(self.x_std))
        y_test_pre = self.standard_tool.inverse_y(self.model.predict(self.standard_tool.transform_x(x_test)))
        if loss:
            self.plot_tool.lossplot(loss,val_loss=val_loss,path=self.model_name)

        if len(y_pre.shape) == 1:
            if len(self.y.shape) > 1:
                self.y = self.y[:,0]
            if len(y_test.shape) > 1:
                y_test = y_test[:,0]
            self.__sub_plot(self.y,y_pre,path=self.model_name+"_train")
            self.__sub_plot(y_test, y_test_pre, path=self.model_name + "_test")
            result_train = pd.DataFrame(np.stack([self.y,y_pre],axis=1),columns=["Actual","Predicted"])
            result_test = pd.DataFrame(np.stack([y_test, y_test_pre], axis=1), columns=["Actual", "Predicted"])
            save_to_excel([(result_train,"train"),(result_test,"test")],path = self.model_name + "actual_predicted")

        else:
            for i in range(y_pre.shape[0]):
                self.__sub_plot(self.y[:,i], y_pre[:,i], path="y{}_".format(i+1) + self.model_name + "_train")
                self.__sub_plot(y_test[:,i], y_test_pre[:,i], path="y{}_".format(i+1) +self.model_name + "_test")
            save_to_excel(pd.DataFrame(np.stack([self.y, y_pre], axis=1), columns=["Actual", "Predicted"]),
                          path="train_actual_predicted")
            save_to_excel(pd.DataFrame(np.stack([y_test, y_test_pre], axis=1), columns=["Actual", "Predicted"]),
                          path="test_actual_predicted")
            result_train_actual = pd.DataFrame(self.y,columns=["y_{}".format(i+1) for i in range(self.y.shape[1])])
            result_train_pre = pd.DataFrame(y_pre, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            result_test_actual = pd.DataFrame(y_test, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            result_test_pre = pd.DataFrame(y_test_pre, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            save_to_excel([(result_train_actual,"train_actual"),(result_train_pre,"train_pre"),(result_test_actual,"test_actual"),(result_test_pre,"test_pre")])

    def __sub_plot(self,y,y_pre,path=None):
        self.plot_tool.actual_pre_line(y, y_pre, path=path)
        self.plot_tool.r2plot(y, y_pre, path=path)

    def save_model(self):
        if "model" not in os.listdir():
            os.mkdir("./model")
        joblib.dump(self.model,"./model/" + "{}.m".format(self.model_name))

class CyrusLR(CyrusModel):
    def __init__(self,*args,**kwargs):
        super(CyrusLR,self).__init__(*args,**kwargs)\


    @CyrusModelMessage("LR")
    def build_model(self):
        self.model_name = "LR"
        self.model = LinearRegression()

    @CyrusModelMessage("LR")
    def fit(self):
        self.model.fit(self.x_std,self.y_std)
        coefs = self.model.coef_
        intercept = self.model.intercept_
        save_to_excel([(pd.DataFrame(coefs[:,None]),"coef"),(pd.DataFrame([intercept]),"intercept")],path="lr_coef_intercept")
        self.save_model()


class CyrusSVR(CyrusModel):
    def __init__(self,*args,**kwargs):
        super(CyrusSVR,self).__init__(*args,**kwargs)

    @CyrusModelMessage("SVR")
    def build_model(self,gamma='scale', C=1.0, epsilon=0.1):
        self.model_name = "SVR"
        self.model = SVR( kernel='rbf',  gamma=gamma,
                  tol=1e-6, C=C, epsilon=epsilon,)

    def hyper_paras_select(self,init_params = {},cv_params = {}):
        """
    kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1,
        """
        """
        :param init_params: {'kernel': 'rbf', 'gamma': 'scale', 'C': 1.0, 'epsilon': 0.1, }
        :param cv_params: 
                          {'gamma': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10,100]}
                          {'epsilon': [0.01, 0.05, 0.1, 0.5, 1]}
                          
        :return:
        """
        model = SVR(**init_params)

        opt_model = GridSearchCV(estimator=model, param_grid=cv_params, cv=5, scoring=["r2", "neg_mean_squared_error"],
                                 refit="r2")
        result = opt_model.fit(self.x_std, self.y_std)
        shape = [len(val) for val in cv_params.values()]
        params = [val for val in cv_params.keys()]
        data = [val for val in cv_params.values()]
        r2_result = opt_model.cv_results_["mean_test_r2"]
        mse_result = -opt_model.cv_results_['mean_test_neg_mean_squared_error']
        if len(cv_params.keys()) == 1:
            self.plot_tool.line_scatter(x=data[0], y=r2_result, x_label=params[0], y_label="r2", path="cv_r2")
            self.plot_tool.line_scatter(x=data[0], y=mse_result, x_label=params[0], y_label="mse", path="cv_mse")
            save_to_excel(pd.DataFrame(r2_result[:, None], index=data[0], columns=["r2"]),
                          path="r2_" + "_".join(params))
            save_to_excel(pd.DataFrame(mse_result[:, None], index=data[0], columns=["mse"]),
                          path="mse_" + "_".join(params))
        else:

            r2_result = r2_result.reshape(shape[::-1])
            mse_result = mse_result.reshape(shape[::-1])
            self.plot_tool.grid_3d_plot(x=data[0], y=data[1], z=r2_result, x_label=params[0], y_label=params[1],
                                        z_label="r2", path="cv_r2")
            self.plot_tool.grid_3d_plot(x=data[0], y=data[1], z=mse_result, x_label=params[0], y_label=params[1],
                                        z_label="mse", path="cv_mse")
            save_to_excel(pd.DataFrame(r2_result, index=data[1], columns=data[0]), path="r2_" + "_".join(params))
            save_to_excel(pd.DataFrame(mse_result, index=data[1], columns=data[0]), path="mse_" + "_".join(params))
        best_params = result.best_params_
        best_score = result.best_score_
        self.logger.info("best params is {}!".format(best_params))
        self.logger.info("best score is {}!".format(best_score))
        return best_params, best_score


class CyrusXGB(CyrusModel):
    def __init__(self,*args,**kwargs):
        super(CyrusXGB,self).__init__(*args,**kwargs)

    @CyrusModelMessage("xgboost")
    def build_model(self,learning_rate=0.1,
                            n_estimators=80,
                            max_depth=2,
                            min_child_weight=8,
                            seed=0,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            gamma=0.1,
                            reg_alpha=0.05,
                            reg_lambda=1,
                            ):
        self.model_name = "xgboost"
        self.model = XGBRegressor(max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        objective='reg:squarederror',
                        booster='gbtree',
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        seed = seed,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,


                       )

    @CyrusModelMessage("xgboost")
    def fit(self):
        self.model.fit(self.x_std,self.y_std)
        importance = self.model.feature_importances_
        save_to_excel(pd.DataFrame(importance[:,np.newaxis]),path="xgb_importance")
        self.save_model()

    @CyrusModelMessage("xgboost")
    def hyper_paras_select(self,init_params = {},cv_params = {}):
        """
        :param init_params: {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
        :param cv_params: {'n_estimators': [100, 200, 300, 400,500,600,700, 800]}
                          {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
                          {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
                          {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
                          {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
                          {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
        :return:
        """
        model = XGBRegressor(**init_params)
        opt_model = GridSearchCV(estimator=model,param_grid=cv_params,cv=5,scoring=["r2","neg_mean_squared_error"],refit="r2")
        result = opt_model.fit(self.x_std,self.y_std)
        shape = [len(val) for val in cv_params.values()]
        params = [val for val in cv_params.keys()]
        data = [val for val in cv_params.values()]
        r2_result = opt_model.cv_results_["mean_test_r2"]
        mse_result = -opt_model.cv_results_['mean_test_neg_mean_squared_error']
        if len(cv_params.keys()) == 1:
            self.plot_tool.line_scatter(x=data[0],y=r2_result,x_label=params[0],y_label="r2",path="cv_r2")
            self.plot_tool.line_scatter(x=data[0], y=mse_result, x_label=params[0], y_label="mse", path="cv_mse")
            save_to_excel(pd.DataFrame(r2_result[:,None],index=data[0],columns=["r2"]),path="r2_"+"_".join(params))
            save_to_excel(pd.DataFrame(mse_result[:, None], index=data[0], columns=["mse"]), path="mse_"+"_".join(params))
        else:

            r2_result = r2_result.reshape(shape[::-1])
            mse_result = mse_result.reshape(shape[::-1])
            self.plot_tool.grid_3d_plot(x=data[0],y=data[1],z=r2_result,x_label=params[0],y_label=params[1],
                                        z_label="r2",path="cv_r2")
            self.plot_tool.grid_3d_plot(x=data[0], y=data[1], z=mse_result, x_label=params[0], y_label=params[1],
                                        z_label="mse", path="cv_mse")
            save_to_excel(pd.DataFrame(r2_result, index=data[1], columns=data[0]), path="r2_" + "_".join(params))
            save_to_excel(pd.DataFrame(mse_result, index=data[1], columns=data[0]), path="mse_" + "_".join(params))
        best_params = result.best_params_
        best_score = result.best_score_
        self.logger.info("best params is {}!".format(best_params))
        self.logger.info("best score is {}!".format(best_score))
        return best_params,best_score

class CyrusNN(CyrusModel):
    def __init__(self, *args, **kwargs):
        super(CyrusNN, self).__init__(*args, **kwargs)

    @CyrusModelMessage("NN")
    def build_model(self, net_structe = [16,20,1]):
        self.model_name = "NN"
        self.model = tf.keras.Sequential()
        for num in net_structe[1:-1]:
            self.model.add(self._hidden_block(num))
        self.model.add(self._output_block(net_structe[-1]))
        self.model.build(input_shape=[None,net_structe[0]])
        optmizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.losses.MeanSquaredError()
        self.model.compile(loss=loss,optimizer=optmizer,metrics=[tf.keras.metrics.mean_squared_error,tf.keras.metrics.mean_absolute_error])
        self.logger.info(self.model.summary())


    @CyrusModelMessage("NN")
    def fit(self,epoch=200,val_data=()):
        if len(self.y_std.shape) == 1:
            self.y_std = self.y_std[:,None]
        self._load_tensorboard()
        callback = tf.keras.callbacks.TensorBoard("./logs/log_loss")

        if val_data:
            val_data = list(val_data)
            val_data[0] = self.standard_tool.transform_x(val_data[0])
            val_data[1] = self.standard_tool.transform_y(val_data[1])
            val_data = tuple(val_data)
            result = self.model.fit(self.x_std,self.y_std,epochs=epoch,validation_data=val_data,
                                    callbacks = [callback])
        else:
            result = self.model.fit(self.x_std,self.y_std,epochs=epoch,callbacks = [callback])

        loss = result.history["loss"]
        save_loss = pd.DataFrame()
        save_loss["loss"] = loss

        if val_data:
            val_loss = result.history["val_loss"]
            save_loss["val_loss"] = val_loss
            self.plot_tool.lossplot(loss,val_loss=val_loss,path="loss_"+self.model_name)
        else:
            self.plot_tool.lossplot(loss,path="loss_"+self.model_name)
        save_to_excel(save_loss,path="loss_{}".format(self.model_name))
        self.save_model()


    def _hidden_block(self,num_ceil):

        return tf.keras.Sequential([
            tf.keras.layers.Dense(num_ceil),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.Activation(tf.nn.leaky_relu)
        ])

    def _output_block(self,num_ceil):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(num_ceil),
            # tf.keras.layers.BatchNormalization(trainable=True),
        ])

    @CyrusModelMessage("NN")
    def save_model(self):
        if "model" not in os.listdir():
            os.mkdir("./model")
        self.model.save("./model/" + "{}.h5".format(self.model_name))

class CyrusNNClass(CyrusNN):
    def __init__(self,*args, **kwargs):
        super(CyrusNNClass, self).__init__(*args, **kwargs)

    @CyrusModelMessage("NNClass")
    def build_model(self, net_structe=[21, 15, 2]):
        self.model_name = "NNClass"
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(net_structe[1]),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.Activation(tf.nn.sigmoid),
            tf.keras.layers.Dense(net_structe[2]),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        self.model.build(input_shape=[None, net_structe[0]])
        optmizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.losses.CategoricalCrossentropy()
        self.model.compile(loss=loss, optimizer=optmizer,
                           metrics=[tf.keras.metrics.categorical_accuracy])
        self.logger.info(self.model.summary())

    @CyrusModelMessage("NNClass")
    def fit(self, epoch=200, val_data=()):

        self._load_tensorboard()
        callback = tf.keras.callbacks.TensorBoard("./logs/log_loss")

        if val_data:
            val_data = list(val_data)
            val_data[0] = self.standard_tool.transform_x(val_data[0])
            val_data[1] = self.standard_tool.transform_y(val_data[1])
            val_data = tuple(val_data)
            result = self.model.fit(self.x_std, self.y_std, epochs=epoch, validation_data=val_data,
                                    callbacks=[callback])
        else:
            result = self.model.fit(self.x_std, self.y, epochs=epoch, callbacks=[callback])

        loss = result.history["loss"]
        save_loss = pd.DataFrame()
        save_loss["loss"] = loss

        if val_data:
            val_loss = result.history["val_loss"]
            save_loss["val_loss"] = val_loss
            self.plot_tool.lossplot(loss, val_loss=val_loss, path="loss_" + self.model_name)
        else:
            self.plot_tool.lossplot(loss, path="loss_" + self.model_name)
        save_to_excel(save_loss, path="loss_{}".format(self.model_name))
        self.save_model()

    def evaluate(self):
        y_pre = self.model.predict(self.x_std)

        self.metric_tool.class_metrics(self.y[:,1],(y_pre[:,1]>=0.5).astype(np.int),y_pre[:,1])


class CyrusAutoEncoder(CyrusNN):
    def __init__(self, add_noise=False,gamma = 0.05,*args, **kwargs):
        super(CyrusAutoEncoder, self).__init__(*args, **kwargs)
        self.gamma = gamma

        if add_noise:
            self.add_gaussian_noise()

    def add_gaussian_noise(self):
        self.x_std += np.random.normal(0,scale=self.gamma,size=self.x_std.shape)

    def custom_loss(self):
        def mse_kl(y_true,y_pred):
            return 0.5 * (tf.losses.mean_squared_error(y_true,y_pred) + tf.losses.kullback_leibler_divergence(y_true,y_pred) )
        return mse_kl

    @CyrusModelMessage("AutoEncoder")
    def build_model(self, net_structe=[256, 64, 32,2,32,64,256]):
        self.model_name = "AutoEncoder"
        self.model = tf.keras.Sequential()
        for num in net_structe[1:-1]:
            self.model.add(self._hidden_block(num))
        self.model.add(self._output_block(net_structe[-1]))
        self.model.build(input_shape=[None, net_structe[0]])
        self.x_in = tf.keras.layers.Input((net_structe[0],))
        optmizer = tf.keras.optimizers.Adam()
        loss = tf.losses.MeanSquaredError()
        # loss = self.custom_loss()
        self.model.compile(loss=loss, optimizer=optmizer,
                           metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])
        self.model_depth = (len(net_structe)//2)
        self.logger.info(self.model.summary())

    @CyrusModelMessage("AutoEncoder")
    def decom_dims(self,data):
        data_decom = self.encoder.predict(self.standard_tool.transform_x(data))
        save_to_excel(pd.DataFrame(data_decom),path="DataAfterDecom")
        self.logger.info("Data has been reduced to 2 dims!")
        return data_decom


    @CyrusModelMessage("AutoEncoder")
    def save_model(self):
        if "model" not in os.listdir():
            os.mkdir("./model")

        layers = self.model.layers
        for i in range(self.model_depth):
            if i == 0:
                x = layers[i](self.x_in)
            else:
                x = layers[i](x)
        self.encoder = tf.keras.models.Model(inputs=self.x_in,
                                             outputs=x)
        self.logger.info(self.encoder.summary())
        self.model.save("./model/" + "all_{}.h5".format(self.model_name))
        self.encoder.save("./model/" + "Encoder_{}.h5".format(self.model_name))


class MultiAttention(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_heads, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.kernel_initializer = kernel_initializer
        super(MultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_heads, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_heads * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        super(MultiAttention, self).build(input_shape)

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])

        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
        e = e / (self.output_dim * 0.5)
        e = K.softmax(e)

        output = K.batch_dot(e, v)

        for i in range(1, self.num_heads):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])

            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
            e = e / (self.output_dim * 0.5)
            e = K.softmax(e)

            o = K.batch_dot(e, v)
            output = K.concatenate(output, o)
        z = K.dot(output, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

class CNSSBlock(tf.keras.Model):
    def __init__(self,fc_struct = [64,32],lstm_struct=[64,32],attention_struct=[64,32],attention_fc__struct=[64],out_struct= [1]):
        super(CNSSBlock,self).__init__()

        self.lstm1 =  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_struct[0],return_sequences=True,unroll=True,))
        self.lstm2 =  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_struct[1],unroll=True))

        self.attention1 = MultiAttention(attention_struct[0],1)
        self.attention2 = MultiAttention(attention_struct[1],1)

        self.fc1 = tf.keras.layers.Dense(fc_struct[0])
        self.fc2 = tf.keras.layers.Dense(fc_struct[1])
        self.fc_attetion = tf.keras.layers.Dense(attention_fc__struct[0])

        self.fc_out = tf.keras.layers.Dense(out_struct[0])

        self.flatten = tf.keras.layers.Flatten()

        self.bn1 = tf.keras.layers.BatchNormalization(trainable=True,center=True,scale=True)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=True,center=True,scale=True)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=True,center=True,scale=True)
        self.bn4 = tf.keras.layers.BatchNormalization(trainable=True,center=True,scale=True)

        self.act1 = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.act2 = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.act3 = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.act4 = tf.keras.layers.Activation(tf.nn.sigmoid)




    def call(self,inputs,training=None):
        x2 = inputs[:,:,0]
        x1 = inputs[:,:,1:]
        x_lstm = self.lstm1(x1)
        x_lstm = self.lstm2(x_lstm)

        x_att = self.attention1(x1)
        x_att = self.attention2(x_att)
        x_att = self.flatten(x_att)
        x_att = self.fc_attetion(x_att)
        x_att = self.bn1(x_att)
        x_att = self.act1(x_att)


        x_fc = self.fc1(x2)
        x_fc = self.bn2(x_fc)
        x_fc = self.act2(x_fc)


        x_fc = self.fc2(x_fc)
        x_fc = self.bn3(x_fc)
        x_fc = self.act3(x_fc)


        x = tf.concat([x_lstm,x_att,x_fc],axis=-1)

        x = self.fc_out(x)
        return x

class CyrusCNSS():
    def __init__(self, x,x_series, y, logger=None):
        if "logs" not in os.listdir():
            os.mkdir("./logs")
        self.logger = logger
        self.metric_tool = CyrusMetrics(logger=self.logger)
        self.plot_tool = PlotTool(self.logger)
        self.x = x
        self.y = y
        self.__standard_data()
        self.mean = x_series.mean(axis=0)
        self.std = x_series.std(axis=0)
        self.x_series_std = (x_series-self.mean)/self.std
        self.x_train = np.concatenate([self.x_std[:,:,None],self.x_series_std],axis=-1)

    def _load_tensorboard(self):
        self.logger.info("start to load tensorboard, please wait!")
        if os.path.exists("./logs/log_loss"):
            shutil.rmtree("./logs/log_loss")
        subprocess.Popen("tensorboard --logdir ./logs/log_loss",shell=False)
        time.sleep(5)
        webbrowser.open("http://localhost:6006")
        self.logger.info("load tensorboard successfully,please click http://localhost:6006 if not jump!")

    def __standard_data(self):
        if len(self.y.shape) > 1:
            if self.y.shape[1] == 1:
                self.y = self.y[:,0]
                self.standard_tool =StandardTool(self.x,self.y,y_dims=1)
            else:
                self.standard_tool = StandardTool(self.x, self.y, y_dims=2)
        else:
            self.standard_tool = StandardTool(self.x, self.y, y_dims=1)
        self.x_std = self.standard_tool.transform_x(self.x)
        self.y_std = self.standard_tool.transform_y(self.y)
        if "pkl_file" not in os.listdir():
            os.mkdir("./pkl_file")
        with open("./pkl_file/standard_tool_CNSS.pkl".format(),"wb") as f:
            pkl.dump(self.standard_tool,f)

    def build_model(self,fc_struct = [10,10],lstm_struct=[10,10],attention_struct=[10,10],attention_fc__struct=[10],out_struct= [1]):
        self.model_name = "CNSS"
        self.model = CNSSBlock(fc_struct = fc_struct,lstm_struct=lstm_struct,attention_struct=attention_struct,\
                          attention_fc__struct=attention_fc__struct,out_struct= out_struct)

        optmizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.losses.MeanSquaredError()
        self.model.compile(loss=loss, optimizer=optmizer,
                           metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.mean_absolute_error])



    def fit(self,epoch=200,val_data=()):
        if len(self.y_std.shape) == 1:
            self.y_std = self.y_std[:,None]
        self._load_tensorboard()
        callback = tf.keras.callbacks.TensorBoard("./logs/log_loss")

        if val_data:

            val_data = list(val_data)
            val_data[0] = np.concatenate([self.standard_tool.transform_x(val_data[0][0])[:,:,None], (val_data[0][1] - self.mean) / self.std], axis=-1)
            val_data[1] = self.standard_tool.transform_y(val_data[1])[:,None]
            val_data = tuple(val_data)
            result = self.model.fit(self.x_train,self.y_std,epochs=epoch,validation_data=val_data,
                                    callbacks = [callback])
        else:
            result = self.model.fit(self.x_train,self.y_std,epochs=epoch,callbacks = [callback])

        loss = result.history["loss"]
        save_loss = pd.DataFrame()
        save_loss["loss"] = loss

        if val_data:
            val_loss = result.history["val_loss"]
            save_loss["val_loss"] = val_loss
            self.plot_tool.lossplot(loss,val_loss=val_loss,path="loss_"+self.model_name)
        else:
            self.plot_tool.lossplot(loss,path="loss_"+self.model_name)
        save_to_excel(save_loss,path="loss_{}".format(self.model_name))


    def predict(self,x=[None,None]):
        x = np.concatenate([self.standard_tool.transform_x(x[0])[:,:,None],(x[1]-self.mean)/self.std],axis=-1)
        return self.standard_tool.inverse_y(self.model.predict(x))

    def evaluate(self,x_test,y_test):

        y_pre = self.standard_tool.inverse_y(self.model.predict(self.x_train))
        x = np.concatenate([self.standard_tool.transform_x(x_test[0])[:,:,None], (x_test[1] - self.mean) / self.std], axis=-1)
        y_test_pre = self.standard_tool.inverse_y(self.model.predict(x))
        metric_train = self.metric_tool.regression_metrics(self.y,y_pre,path=self.model_name + "_train")
        metric_test = self.metric_tool.regression_metrics(y_test,y_test_pre,path=self.model_name + "_test")


        self.logger.info("metrics for train:")
        self.logger.info(metric_train)
        self.logger.info("metrics for test:")
        self.logger.info(metric_test)

    def plot_pre(self,x_test,y_test,loss=None,val_loss=None):
        y_pre = self.standard_tool.inverse_y(self.model.predict(self.x_train))
        x = np.concatenate([self.standard_tool.transform_x(x_test[0])[:,:,None], (x_test[1] - self.mean) / self.std], axis=-1)
        y_test_pre = self.standard_tool.inverse_y(self.model.predict(x))
        if loss:
            self.plot_tool.lossplot(loss,val_loss=val_loss,path=self.model_name)

        if len(y_pre.shape) == 1:
            if len(self.y.shape) > 1:
                self.y = self.y[:,0]
            if len(y_test.shape) > 1:
                y_test = y_test[:,0]
            self.__sub_plot(self.y,y_pre,path=self.model_name+"_train")
            self.__sub_plot(y_test, y_test_pre, path=self.model_name + "_test")
            result_train = pd.DataFrame(np.stack([self.y,y_pre],axis=1),columns=["Actual","Predicted"])
            result_test = pd.DataFrame(np.stack([y_test, y_test_pre], axis=1), columns=["Actual", "Predicted"])
            save_to_excel([(result_train,"train"),(result_test,"test")],path = self.model_name + "actual_predicted")

        else:
            for i in range(y_pre.shape[0]):
                self.__sub_plot(self.y[:,i], y_pre[:,i], path="y{}_".format(i+1) + self.model_name + "_train")
                self.__sub_plot(y_test[:,i], y_test_pre[:,i], path="y{}_".format(i+1) +self.model_name + "_test")
            save_to_excel(pd.DataFrame(np.stack([self.y, y_pre], axis=1), columns=["Actual", "Predicted"]),
                          path="train_actual_predicted")
            save_to_excel(pd.DataFrame(np.stack([y_test, y_test_pre], axis=1), columns=["Actual", "Predicted"]),
                          path="test_actual_predicted")
            result_train_actual = pd.DataFrame(self.y,columns=["y_{}".format(i+1) for i in range(self.y.shape[1])])
            result_train_pre = pd.DataFrame(y_pre, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            result_test_actual = pd.DataFrame(y_test, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            result_test_pre = pd.DataFrame(y_test_pre, columns=["y_{}".format(i + 1) for i in range(self.y.shape[1])])
            save_to_excel([(result_train_actual,"train_actual"),(result_train_pre,"train_pre"),(result_test_actual,"test_actual"),(result_test_pre,"test_pre")])

    def __sub_plot(self,y,y_pre,path=None):
        self.plot_tool.actual_pre_line(y, y_pre, path=path)
        self.plot_tool.r2plot(y, y_pre, path=path)



if __name__ == '__main__':
    import logging
    import sys
    logger = logging.getLogger("model")
    logger.setLevel(logging.INFO)
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    data = np.random.randn(1000,256)
    model = CyrusAutoEncoder(add_noise=True,logger=logger,x=data,y=data)
    model.build_model()
    model.fit(epoch=10)
    model.save_model()