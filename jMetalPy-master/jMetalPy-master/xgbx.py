

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import random
from xgboost import plot_importance
from sklearn.impute import SimpleImputer

class model(object):
    mo=xgb.XGBRegressor()
    def __init__(self):
        trainFilePath = '/home/zionxie/sl/train.csv'
        testFilePath = '/home/zionxie/sl/test.csv'

        model.mo = self.trainmodel(trainFilePath, testFilePath)

    def loadDataset(self,filePath):
        df = pd.read_csv(filepath_or_buffer=filePath)
        return df


    def featureSet(self,data):
        data_num = len(data)
       # print (data)
        XList = []
        yList = []
        for row in range(0, data_num):
            tmp_list = []
            tmp_list.append(data.iloc[row]['1'])
            tmp_list.append(data.iloc[row]['2'])
            tmp_list.append(data.iloc[row]['3'])
            tmp_list.append(data.iloc[row]['4'])
            tmp_list.append(data.iloc[row]['5'])
            tmp_list.append(data.iloc[row]['6'])
            tmp_list.append(data.iloc[row]['7'])
            tmp_list.append(data.iloc[row]['8'])
            tmp_list.append(data.iloc[row]['9'])
            tmp_list.append(data.iloc[row]['10'])
            tmp_list.append(data.iloc[row]['11'])
            tmp_list.append(data.iloc[row]['12'])
            tmp_list.append(data.iloc[row]['13'])
            tmp_list.append(data.iloc[row]['14'])
            tmp_list.append(data.iloc[row]['15'])
            tmp_list.append(data.iloc[row]['16'])
            tmp_list.append(data.iloc[row]['17'])
            tmp_list.append(data.iloc[row]['18'])
            tmp_list.append(data.iloc[row]['19'])
            tmp_list.append(data.iloc[row]['20'])
            tmp_list.append(data.iloc[row]['21'])
            tmp_list.append(data.iloc[row]['22'])
            tmp_list.append(data.iloc[row]['23'])
            tmp_list.append(data.iloc[row]['24'])
            tmp_list.append(data.iloc[row]['25'])
            tmp_list.append(data.iloc[row]['26'])
            tmp_list.append(data.iloc[row]['27'])
            tmp_list.append(data.iloc[row]['28'])
            tmp_list.append(data.iloc[row]['29'])
            tmp_list.append(data.iloc[row]['30'])
            tmp_list.append(data.iloc[row]['31'])
           # tmp_list.append(data.iloc[row]['32'])
            yList.append(data.iloc[row]['32'])
            XList.append(tmp_list)
        print(yList)
        #yList = data.y.values


       # print(yList)
        return XList, yList


    def loadTestData(self,filePath):
        data = pd.read_csv(filepath_or_buffer=filePath)
        data_num = len(data)
        XList = []
        for row in range(0, 1):
            tmp_list = []
            tmp_list.append(data.iloc[row]['1'])
            tmp_list.append(data.iloc[row]['2'])
            tmp_list.append(data.iloc[row]['3'])
            tmp_list.append(data.iloc[row]['4'])
            tmp_list.append(data.iloc[row]['5'])
            tmp_list.append(data.iloc[row]['6'])
            tmp_list.append(data.iloc[row]['7'])
            tmp_list.append(data.iloc[row]['8'])
            tmp_list.append(data.iloc[row]['9'])
            tmp_list.append(data.iloc[row]['10'])
            tmp_list.append(data.iloc[row]['11'])
            tmp_list.append(data.iloc[row]['12'])
            tmp_list.append(data.iloc[row]['13'])
            tmp_list.append(data.iloc[row]['14'])
            tmp_list.append(data.iloc[row]['15'])
            tmp_list.append(data.iloc[row]['16'])
            tmp_list.append(data.iloc[row]['17'])
            tmp_list.append(data.iloc[row]['18'])
            tmp_list.append(data.iloc[row]['19'])
            tmp_list.append(data.iloc[row]['20'])
            tmp_list.append(data.iloc[row]['21'])
            tmp_list.append(data.iloc[row]['22'])
            tmp_list.append(data.iloc[row]['23'])
            tmp_list.append(data.iloc[row]['24'])
            tmp_list.append(data.iloc[row]['25'])
            tmp_list.append(data.iloc[row]['26'])
            tmp_list.append(data.iloc[row]['27'])
            tmp_list.append(data.iloc[row]['28'])
            tmp_list.append(data.iloc[row]['29'])
            tmp_list.append(data.iloc[row]['30'])
            tmp_list.append(data.iloc[row]['31'])
          #  tmp_list.append(data.iloc[row]['32'])
            XList.append(tmp_list)
        print(XList)
        return XList

    def trainmodel(self,trainpath: object, testpath: object) -> object:
        data = self.loadDataset(trainpath)
        X_train, y_train = self.featureSet(data)
        X_test = self.loadTestData(testpath)
        x_np = np.array(X_train)
        y_np = np.array(y_train)
        t_np = np.array(X_test)
        model= self.trainandTest(x_np, y_np, t_np)
        return model

    @classmethod
    def function(self,data):



        ans = self.mo.predict(data)

        return ans

    def trainandTest(self,X_train, y_train, X_test):



        print(X_train.shape )
        print(y_train.shape )

        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
        model.fit(X_train, y_train)


        return model


      #  pd_data = pd.DataFrame(np_data, columns=['id', '32'])
      #  print(pd_data)
      #  pd_data.to_csv('/home/zionxie/sl/data.csv', index=None,encoding='gbk')


       # plot_importance(model)
        #plt.show()

if __name__ == '__main__':
    """
    trainFilePath = '/home/zionxie/sl/train.csv'
    testFilePath = '/home/zionxie/sl/test.csv'


    model=mo.trainmodel(trainFilePath,testFilePath)
    """

    m=model()

    num = range(0, 100)  # 范围在0到100之间，需要用到range()函数。
    data = random.sample(num, 31)  # 选取10个元素
    print(data)
    tmp_list = []
    tmp_list.append(data)
    data_n = np.array(tmp_list)
    res=m.function(data_n)
    print(res)