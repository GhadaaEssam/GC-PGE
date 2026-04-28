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
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score,confusion_matrix,accuracy_score
from sklearn import model_selection
import numpy as np
from model.preprocess import make_data_geo, get_train_edge, make_data, pgb
from scipy.special import erfinv 
from model.model import Model


EPSILON = np.finfo(float).eps


def get_metrics(out_, edge_label_):
    out = out_.detach().cpu().numpy()
    edge_label = edge_label_.detach().cpu().numpy()

    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    accuracy = accuracy_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap,accuracy

def test(model,data,data_geo):
    model.eval()
    target = data.y
    # out,prediction,_ ,_,_,temp= model(data,data_geo.X_test)
    result= model(data,data_geo.X_test)
    out = result['out']
    prediction=result['vimp_g']
    temp=result['temp']
    cor=result['cor']

    out = out.max(dim=1).indices
    temp = temp.max(dim=1).indices

    result_= model(data,data_geo.X_train)
    out_train = result_['out'].max(dim=1).indices
    
    auc1, f1, ap ,_= get_metrics(prediction[data.test_mask], target[data.test_mask])
    auc,_,_,_ = get_metrics(cor[data.test_mask], target[data.test_mask])
    auc_geo,f1_geo,ap_geo,acc_geo = get_metrics(out,data_geo.Y_test)
    auc_temp,_,_ ,_= get_metrics(temp,data_geo.Y_test)
    auc_geo_train,_,_,_ = get_metrics(out_train,data_geo.Y_train)
    
    model.train()
    # return auc, f1, ap, auc_geo,auc_geo_train,auc_temp,f1_geo,ap_geo
    return {'auc':auc,'f1':f1,'ap':ap,'auc_geo':auc_geo,'auc_geo_train':auc_geo_train,'auc_temp':auc_temp,'f1_geo':f1_geo,'ap_geo':ap_geo,'acc_geo':acc_geo,'cor':auc1}

def train_model(data_geo, label_geo, anchor_list, data_x, data_ppi_link_index, data_homolog_index,progressBarObj):
    
    if os.path.exists('result/'):
        pass
    else:
        os.mkdir('result/')
        os.mkdir('result/model/')

    rankGauss = (data_geo.values/data_geo.values.max()-0.5)*2
    rankGauss = np.clip(rankGauss, -1+EPSILON, 1-EPSILON)
    rankGauss = erfinv(rankGauss) 
    data_geo = pd.DataFrame(rankGauss,columns=data_geo.columns)
    data_geo_obj = make_data_geo(data_geo, label_geo,10,4,4709)

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor,test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
    test_anchor_csv=pd.DataFrame(test_anchor,dtype=int)
    test_anchor_csv.to_csv(r'result/test_anchor.csv')

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))

    pgb1 = pgb(progressBarObj,0,20)
    train_edge_ppi , _ = get_train_edge(data_ppi_link_index, train_anchor,pgb1)
    
    pgb2 = pgb(progressBarObj,20,40)
    train_edge_homolog , _ = get_train_edge(data_homolog_index, train_anchor,pgb2)
    
    data_obj = make_data(data_x,train_edge_ppi,train_edge_homolog,anchor_list,test_anchor)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    df_acc = pd.DataFrame(columns=('epoch','auc_geo','auc_train','auc','loss'))
    
    my_net = Model(data_geo_x_shape=data_geo_obj.X_train.shape,num_muti_gat=8,num_muti_mlp=5,num_node_features=data_obj.num_node_features,data_x_N=data_obj.train_mask.shape[0])
    # my_net = GraphCNN(in_c=data_obj.num_node_features, hid_c=8, out_c=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检查设备
    my_net = my_net.to(device)  
    data = data_obj.to(device)  
    data_geo_obj = data_geo_obj.to(device)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.005)  # 优化器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.93303)
    alpha = 0.5
    auc_stock = 0.0
    num = 10
    my_net.train()
    epoches = 500

    pgb3 = pgb(progressBarObj,40,98)

    for epoch in range(epoches):
        optimizer.zero_grad()

        # np.random.seed(seed+i)
        lam = np.random.beta(alpha, alpha)
        # torch.manual_seed(seed+i)
        index = torch.randperm(data_geo_obj.X_train.size(0)).cpu()
        torch.seed()
        mixed_x = lam * data_geo_obj.X_train + (1 - lam) * data_geo_obj.X_train[index, :]

        result = my_net(data,mixed_x)  # 预测结果
        out=result['out']
        loss_mutiGAT=result['loss_mutiGAT']
        loss_L1=result['loss_L1']
        pw_w = result['pw_w']
        loss =   0.1*loss_mutiGAT +0.1*torch.mean(torch.pow(pw_w,2))+0.1*loss_L1 + 1.0*(lam * F.nll_loss(out, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out, data_geo_obj.Y_train[index].long()))


        # out,_,loss_mutiGAT,loss_L1,_ = my_net(data,data_geo_obj.X_train)  # 预测结果
        # loss =  0.1*loss_mutiGAT +0.1*loss_L1 + 1.0*F.nll_loss(out, data_geo_obj.Y_train.long())

        loss.backward()
        optimizer.step()  # 优化器
        # scheduler.step()
        test_= test(my_net, data,data_geo_obj)
        print("epoch:{},auc_geo:{},auc_train:{},auc:{},cor:{},ap:{},loss:{},auc_temp:{},num:{}".format(epoch + 1,test_['auc_geo'],test_['auc_geo_train'], test_['auc'], test_['cor'], test_['ap'] , loss.item(),test_['auc_temp'],num))
        df_acc=df_acc.append(pd.DataFrame({'epoch':[epoch],'auc_geo':[test_['auc_geo']],'auc_train':[test_['auc_geo_train']],'f1_geo':[test_['f1_geo']],'ap_geo':test_['ap_geo'],'auc':[test_['auc']],'cor':[test_['cor']],'loss':[loss.item()],'auc_temp':[test_['auc_temp']],'acc_geo':[test_['acc_geo']]}),ignore_index=True)
        pgb3.update((epoch+1)/epoches)
        
        # if  test_['auc_geo_train'] > 0.99 and epoch>=235: #and epoch>=250 
        #     num = num - 1
        # else:
        #     num = 10
        # if (num == 0 and auc_stock == test_['auc_geo_train']):
        #     print("###")
        #     break    
        
        if (auc_stock <= test_['auc_geo_train']) and test_['auc_geo_train'] > 0.99 : #and epoch>=250 
            num = num - 1
            auc_stock = test_['auc_geo_train']
        else:
            num = 5
            auc_stock = test_['auc_geo_train']
        if (num == 0 and auc_stock <= test_['auc_geo_train']):
            print("###")
            break

    my_net.eval()

    torch.save(my_net,"result/model.pt")
    result = my_net(data,data_geo_obj.X_test)
    pd.DataFrame({"predict":result['cor'].detach().cpu()}).to_csv("result/predict_muti_all.csv",index=False)
    pd.DataFrame({"predict":result['out'].max(dim=1).indices.detach().cpu()}).to_csv("result/predict_out.csv",index=False)
    pd.DataFrame(result['graph'].detach().cpu().numpy()).to_csv("result/graph.csv")
    pd.DataFrame({"predict":result['pw_w'].detach().cpu()}).to_csv("result/pw_w.csv",index=False)
    df_acc.to_csv("result/lossAndAcc.csv")

    progressBarObj.setValue(int(100))