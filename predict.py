import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score,confusion_matrix
from sklearn import model_selection
import numpy as np
from model.preprocess import make_data_geo, get_train_edge, make_data,pgb
from model.model import Model
from scipy.special import erfinv 
EPSILON = np.finfo(float).eps

def make_data_geo_no_label(data_geo):
    data = Data()
    
    data.X = torch.tensor(data_geo.values,dtype=torch.float)
    return data

def predict_model(model_path, data_geo, anchor_list, data_x, data_ppi_link_index, data_homolog_index,progressBarObj):
    print("loading model")
    
    rankGauss = (data_geo.values/data_geo.values.max()-0.5)*2
    rankGauss = np.clip(rankGauss, -1+EPSILON, 1-EPSILON)
    rankGauss = erfinv(rankGauss) 
    data_geo = pd.DataFrame(rankGauss,columns=data_geo.columns)

    data_geo_obj = make_data_geo_no_label(data_geo)

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor,test_anchor = model_selection.train_test_split(anchor_index, test_size=0.2)
    test_anchor_csv=pd.DataFrame(test_anchor,dtype=int)
    test_anchor_csv.to_csv(r'result/test_anchor.csv')

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))

    print("loading ppi network")
    pgb1 = pgb(progressBarObj,0,70)
    train_edge_ppi , _ = get_train_edge(data_ppi_link_index, train_anchor,pgb1)
    
    print("loading homolog network")
    pgb2 = pgb(progressBarObj,70,100)
    train_edge_homolog , _ = get_train_edge(data_homolog_index, train_anchor,pgb2)
    

    data_obj = make_data(data_x,train_edge_ppi,train_edge_homolog,anchor_list,test_anchor)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    my_net = torch.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检查设备
    my_net = my_net.to(device)  
    data = data_obj.to(device)  
    data_geo_obj = data_geo_obj.to(device)
    
    my_net.eval()

    result = my_net(data,data_geo_obj.X)

    return {"out":result['out'].max(dim=1).indices.detach().cpu(),"vimp":result['cor'].detach().cpu(),"graph":result['graph'].detach().cpu().numpy(),"pw_w":result['pw_w'].detach().cpu()}
    # #分类预测
    # out
    # #靶点预测
    # vimp
    # #图预测
    # graph
    # #通路排序
    # pw_w