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
from scipy.special import erfinv 
from sklearn.model_selection import StratifiedKFold


if os.path.exists('result/'):
    pass
else:
    os.mkdir('result/')
    os.mkdir('result/model/')

EPSILON = np.finfo(float).eps

#111
data_geo = pd.read_csv(r"./验证队列/肝癌索拉菲尼/肝癌索拉菲尼耐药result_test_rank.csv", header= 0).iloc[:,1:]
label_geo = pd.read_csv(r"./验证队列/肝癌索拉菲尼/sample.csv",header=0).iloc[:,1]

# data_geo = pd.read_csv(r"./验证队列/卵巢癌顺铂耐药/卵巢癌顺铂耐药result_test_rank.csv", header= 0).iloc[:,1:]
# label_geo = pd.read_csv(r"./验证队列/卵巢癌顺铂耐药/卵巢癌顺铂耐药sample_list.csv",header=0).iloc[:,1]

# data_geo = pd.read_csv(r"./验证队列/免疫治疗/免疫geo表达谱result_test_rank - 副本.csv", header= 0).iloc[:,1:]
# label_geo = pd.read_csv(r"./验证队列/免疫治疗/sample_list - 副本.csv",header=0).iloc[:,1]

# data_geo = pd.read_csv(r"./验证队列/6.乳腺癌靶向曲妥珠单抗/乳腺癌靶向曲妥珠单抗result_test_rank.csv", header= 0).iloc[:,1:]
# label_geo = pd.read_csv(r"./验证队列/6.乳腺癌靶向曲妥珠单抗/sample.csv",header=0).iloc[:,1]

# data_geo = pd.read_csv(r"./验证队列/4.结直肠癌FOLFOX/结直肠癌FOLFOX_result_test_rank.csv", header= 0).iloc[:,1:]
# label_geo = pd.read_csv(r"./验证队列/4.结直肠癌FOLFOX/sample.csv",header=0).iloc[:,1]

# data_geo = pd.read_csv(r"./验证队列/5.乳腺癌TFAC/乳腺癌TFAC_result_test_rank.csv", header= 0).iloc[:,1:]
# label_geo = pd.read_csv(r"./验证队列/5.乳腺癌TFAC/sample.csv",header=0).iloc[:,1]

#################################################################################################################################

rankGauss = (data_geo.values/data_geo.values.max()-0.5)*2
rankGauss = np.clip(rankGauss, -1+EPSILON, 1-EPSILON)
rankGauss = erfinv(rankGauss) 
data_geo = pd.DataFrame(rankGauss,columns=data_geo.columns)

# data_geo.to_csv(r"./result/data_geo.csv")
def make_data_geo(data_geo, label_geo,k,i,seed):
    assert k > 1
    
    data = Data()
    np.random.seed(seed)
    indices = np.random.permutation(range(len(label_geo)))
    # X = data_geo.loc[indices]
    # Y = label_geo.loc[indices]
    X = torch.tensor(data_geo.loc[indices].values,dtype=torch.float)
    Y = torch.tensor(label_geo.loc[indices].values,dtype=torch.int)

    fold_size = X.shape[0] // k
    X_train, Y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], Y[idx]
        if j == i: ###第i折作valid
            X_test, Y_test = X_part, y_part
        elif X_train is None:
            X_train, Y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            Y_train = torch.cat((Y_train, y_part), dim=0)


    data.X_train= X_train
    data.X_test= X_test
    data.Y_train= Y_train
    data.Y_test= Y_test

    # pd.DataFrame(X_test).to_csv(r'result/X_test.csv')
    # pd.DataFrame(Y_test).to_csv(r'result/Y_test.csv')
    return data



#222
anchor_list = pd.read_csv(r"./验证队列/肝癌索拉菲尼/pubmed_result.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/卵巢癌顺铂耐药/卵巢癌顺铂耐药pubmed_result.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/免疫治疗/pubmed_result.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/免疫治疗/pubmed_result - 副本.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/6.乳腺癌靶向曲妥珠单抗/pubmed_result.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/4.结直肠癌FOLFOX/pubmed_result.csv", header= 0)
# anchor_list = pd.read_csv(r"./验证队列/5.乳腺癌TFAC/pubmed_result.csv", header= 0)


################################################################################################
anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
train_anchor,test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
test_anchor_csv=pd.DataFrame(test_anchor,dtype=int)
test_anchor_csv.to_csv(r'result/test_anchor.csv')

# test_anchor = pd.read_csv(r'setting/肝癌test_anchor1.csv', header= 0).iloc[:,1]
# test_anchor = pd.read_csv(r'setting/卵巢癌test_anchor1.csv', header= 0).iloc[:,1]
# # test_anchor = pd.read_csv(r'setting/黑色素瘤test_anchor.csv', header= 0).iloc[:,1]
###############################################################################################



#333

data_x = pd.read_csv('./验证队列/肝癌索拉菲尼/data_x_all.csv',header=0).iloc[:,1:]
data_ppi_link_index = pd.read_csv('./验证队列/肝癌索拉菲尼/ppi_link_hcc_sorafineb_400.csv',header=0)
data_homolog_index = pd.read_csv('./验证队列/肝癌索拉菲尼/homolog_肝癌索拉菲尼.csv',header=0)



# data_x = pd.read_csv('./验证队列/卵巢癌顺铂耐药/data_x_all.csv',header=0).iloc[:,1:]
# data_ppi_link_index = pd.read_csv('./验证队列/卵巢癌顺铂耐药/ppi_link_卵巢顺铂_600.csv',header=0)
# data_homolog_index = pd.read_csv('./验证队列/卵巢癌顺铂耐药/homolog_卵巢顺铂.csv',header=0)



# data_x = pd.read_csv('./验证队列/免疫治疗/data_x_all.csv',header=0).iloc[:,1:]
# data_ppi_link_index = pd.read_csv('./验证队列/免疫治疗/ppi_link_黑色素瘤免疫治疗_600.csv',header=0)
# data_homolog_index = pd.read_csv('./验证队列/免疫治疗/homolog_黑色素瘤免疫治疗.csv',header=0)



# data_x = pd.read_csv('./验证队列/6.乳腺癌靶向曲妥珠单抗/data_x_all.csv',header=0).iloc[:,1:]
# data_ppi_link_index = pd.read_csv('./验证队列/6.乳腺癌靶向曲妥珠单抗/ppi_link_乳腺癌靶向曲妥珠单抗_600.csv',header=0)
# data_homolog_index = pd.read_csv('./验证队列/6.乳腺癌靶向曲妥珠单抗/homolog_乳腺癌靶向曲妥珠单抗.csv',header=0)



# data_x = pd.read_csv('./验证队列/4.结直肠癌FOLFOX/data_x_all.csv',header=0).iloc[:,1:]
# data_ppi_link_index = pd.read_csv('./验证队列/4.结直肠癌FOLFOX/ppi_link_结直肠癌FOLFOX_600.csv',header=0)
# data_homolog_index = pd.read_csv('./验证队列/4.结直肠癌FOLFOX/homolog_结直肠癌FOLFOX.csv',header=0)


# data_x = pd.read_csv('./验证队列/5.乳腺癌TFAC/data_x_all.csv',header=0).iloc[:,1:]
# data_ppi_link_index = pd.read_csv('./验证队列/5.乳腺癌TFAC/ppi_link_乳腺癌TFAC_600.csv',header=0)
# data_homolog_index = pd.read_csv('./验证队列/5.乳腺癌TFAC/homolog_乳腺癌TFAC.csv',header=0)


###########################################

def get_train_edge(data_edge_index, train_anchor):
    train_edge_index = pd.DataFrame(dtype=int)
    test_edge_index = pd.DataFrame(dtype=int)

    for i in range(len(data_edge_index)):
        if(i%10000 == 0):
            print(i/len(data_edge_index))
        if (data_edge_index.iloc[i,0] in train_anchor.values) and (data_edge_index.iloc[i,1] in train_anchor.values):
            train_edge_index = train_edge_index.append(data_edge_index.iloc[i,:])
        elif(data_edge_index.iloc[i,0] in train_anchor.values) or (data_edge_index.iloc[i,1] in train_anchor.values):
            test_edge_index = test_edge_index.append(data_edge_index.iloc[i,:])
    return train_edge_index , test_edge_index
anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))


# train_edge_cor , _ = get_train_edge(data_edge_bg_index, train_anchor)

train_edge_ppi , _ = get_train_edge(data_ppi_link_index, train_anchor)

train_edge_homolog , _ = get_train_edge(data_homolog_index, train_anchor)



def make_data(data_x,data_ppi_link_index,data_homolog_index,anchor_list,test_anchor):
    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    not_anchor_index = anchor_list.result_num[anchor_list.result_num==0].index

    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))
    not_train_anchor = pd.Series(list(set(anchor_list.index)-set(train_anchor.to_list())))

    data_y = pd.Series(0,index=data_x.index,dtype=int)
    data_y[anchor_index.to_list()]=1

    # test_sample = random.sample(not_anchor_index.to_list(),len(anchor_index))
    test_sample = random.sample(not_train_anchor.to_list(),len(train_anchor))

    data_train_mask = pd.Series(False,index=data_x.index,dtype=bool)
    data_train_mask[train_anchor.to_list()]=True
    data_train_mask[test_sample]=True

    # data_test_mask = pd.Series(False,index=data_x.index,dtype=bool)
    # data_test_mask[test_anchor.to_list()]=True
    # data_test_mask[test_sample[len(train_anchor):]]=True
    data_test_mask = pd.Series(True,index=data_x.index,dtype=bool)
    data_test_mask[data_train_mask]=False
    
    



    data = Data()
    data.num_nodes = len(data_x)
    data.num_node_features = data_x.shape[1]
    data.edge_index = {
                       'ppi':torch.tensor(data_ppi_link_index.T.values,dtype=torch.long),
                       'homolog':torch.tensor(data_homolog_index.T.values,dtype=torch.long)
                       }
    

    data.x = torch.tensor(data_x.values,dtype=torch.float)
    data.y = torch.tensor(data_y.values,dtype=torch.int)
    data.train_mask = torch.tensor(data_train_mask.values,dtype=torch.bool)
    data.test_mask = torch.tensor(data_test_mask.values,dtype=torch.bool)
    return data

# data_obj = make_data(data_x,data_edge_bg_index,anchor_list,test_anchor)
data_obj = make_data(data_x,train_edge_ppi,train_edge_homolog,anchor_list,test_anchor)

class MutiGAT(nn.Module):
    def __init__(self,num_muti_gat,num_node_features, hid_c, out_c,data_x_N):
        super(MutiGAT,self).__init__()
        self.num_muti_gat = num_muti_gat
        # self.num_muti_graph = num_muti_graph
        self.data_x_N = data_x_N
        self.gat_list = []
        self.edge_type_list = ["ppi","homolog"]
        # self.edge_type_list = ["cor","ppi","homolog"]
        # self.graph_list= []
        for i in range(num_muti_gat):
            self.gat_list.append(GraphCNN(in_c=num_node_features, hid_c=hid_c, out_c=out_c))
        # for i in range(num_muti_graph):
        #     self.graph_list.append(GraphCNN_Generalization(in_c=num_node_features, hid_c=hid_c, out_c=out_c,data_x_N=data_x_N))
        
        self.gat_list = nn.ModuleList(self.gat_list)
        # self.graph_list = nn.ModuleList(self.graph_list)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.generalization = GeneralizationGraph(embedding_dim = 4,data_x_shape = data_x_N,num_node_features=num_node_features)
    def forward(self,data):
        out = torch.zeros_like(data.y)
        loss = torch.Tensor([0.]).to(next(self.parameters()).device)
        i = 0
        # graph_loss = torch.Tensor([0.]).to(next(self.parameters()).device)
        # graph = torch.zeros(self.data_x_N,self.data_x_N,dtype=torch.float32).to(next(self.parameters()).device)

        for module in self.gat_list:
            temp,temp_loss = module(data,self.edge_type_list[(i%2)])##########
            i=i+1
            out = out + temp[:,1].exp()
            
            loss = loss + temp_loss

        out_graph,graph,pw_w = self.generalization(data.x,out/(self.num_muti_gat))
        
        return out/(self.num_muti_gat), loss/self.num_muti_gat, out_graph[:,1], graph,pw_w


class GraphCNN_Generalization(nn.Module):
    def __init__(self, in_c, hid_c, out_c,data_x_N):
        super(GraphCNN_Generalization, self).__init__()  # 表示子类GraphCNN继承了父类nn.Module的所有属性和方法
        # self.dfr = ConcreteDropout(in_c,temp= 1.0/10.0)
        self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c, dropout=0.6 ,heads=1, concat=False)
        self.bn1   = nn.BatchNorm1d(hid_c)
        self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c, dropout=0.6, heads=1, concat=False)
        self.generalization = GeneralizationGraph(embedding_dim = 4,data_x_shape = data_x_N,num_node_features=in_c)
    def forward(self, data):
        # data.x  data.edge_index
        x = data.x  # [N, C], C为特征的维度
        # x = self.dfr(x)
        edge_index = data.edge_index  # [2, E], E为边的数量
        x = F.dropout(x, p=0.6, training=self.training)
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D], N是节点数量，D是第一层输出的隐藏层的维度
        hid = self.bn1(hid)
        hid = F.leaky_relu(hid)
        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c], out_c就是定义的输出
        out = F.log_softmax(out, dim=1)  # [N, out_c],表示输出
        out1,graph = self.generalization(data.x,out[:,1].exp())
        # loss = 0
        # mask = torch.rand(data.train_mask.shape[0])<0.5
        # loss = F.nll_loss(out1[mask],(out[:,1].exp()>=0.5).to(dtype=torch.long)[mask])
        loss1 = F.nll_loss(out[data.train_mask],data.y[data.train_mask].long())
        return out1,loss1,graph


class GeneralizationGraph(nn.Module):
    def __init__(self,embedding_dim,data_x_shape,num_node_features):
        super(GeneralizationGraph, self).__init__()
        self.embedding_w = nn.Parameter(torch.zeros(num_node_features))
        # self.embedding_vec = nn.Parameter(torch.zeros(data_x_shape,embedding_dim))
        self.dfr = ConcreteDropout(num_node_features)
    def forward(self,data,node_p):
        embedding_data,pw_vimp = self.dfr(data)
        embedding_w = torch.sigmoid(self.embedding_w)
        expanded_data = embedding_data*embedding_w
        
        # expanded_data = torch.cat((embedding_data,self.embedding_vec),dim=1)
        
        # graph_raw = torch.mm(data,data.t())/(data.sum(dim=1)+1e-6)
        graph = torch.mm(expanded_data,expanded_data.t())/(expanded_data.sum(dim=1)+1e-6)
        expanded_node_p = node_p.reshape(node_p.shape[0],1)
        adj_node_p = torch.mm(graph,expanded_node_p)
        adj_node_p_nag = torch.mm(graph,1-expanded_node_p)
        node_p_combanded = torch.cat((adj_node_p_nag,adj_node_p),dim=1)
        norm_node_p = F.softmax(node_p_combanded,dim=1)
        return norm_node_p,graph,pw_vimp

        

class ConcreteDropout(nn.Module):
    def __init__(self,shape,temp= 1.0/10.0):
        super().__init__()
        self.logit_p = nn.Parameter(torch.zeros(shape))
        self.temp = temp
    def forward(self,x):
        if self.training:
            unif_noise = torch.rand_like(self.logit_p)
            # unif_noise = torch.full_like(self.logit_p, 0.5)
        else:
            unif_noise = torch.full_like(self.logit_p, 0.5)
        dropout_p = torch.sigmoid(self.logit_p)
        approx = (
            torch.log(dropout_p + EPSILON)
            - torch.log(1. - dropout_p + EPSILON)
            + torch.log(unif_noise + EPSILON)
            - torch.log(1. - unif_noise + EPSILON)
        )
        approx_output = torch.sigmoid(approx / self.temp)


        return x*(1 - approx_output), (1-dropout_p)
    

class VariableDropoutMLP(nn.Module):
    def __init__(self,data_x_shape,temp= 1.0/10.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_x_shape,1000),  #3000
            nn.BatchNorm1d(1000),
            nn.Dropout(0.9),#0.9
            nn.LeakyReLU(),
            nn.Linear(1000,500),#1000
            nn.BatchNorm1d(500),
            nn.Dropout(0.9),#0.9
            nn.LeakyReLU(),
            nn.Linear(500,300),#300
            nn.BatchNorm1d(300),
            nn.Dropout(0.8),#0.8
            nn.LeakyReLU(),
            nn.Linear(300,2),
            
        )
        self.temp = temp
    def forward(self,x,vimp):
        if self.training:
            unif_noise = torch.rand_like(vimp)
            # unif_noise = torch.full_like(vimp, 0.5)
        else:
            unif_noise = torch.full_like(vimp, 0.5)
        
        approx = (
            torch.log(vimp + EPSILON)
            - torch.log(1-vimp + EPSILON)
            + torch.log(unif_noise + EPSILON)
            - torch.log(1. - unif_noise + EPSILON)
        )
        approx_output = torch.sigmoid(approx / self.temp)

        # out = self.model(x*torch.ones_like(approx_output))
        out = self.model(x*( approx_output))
        out = F.log_softmax(out, dim=1)
        # loss = F.nll_loss(out, y.long())
        # output = out.max(dim=1).indices
        return out


# create the graph cnn model
class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GraphCNN, self).__init__()  # 表示子类GraphCNN继承了父类nn.Module的所有属性和方法
        # self.dfr = ConcreteDropout(in_c,temp= 1.0/10.0)
        self.conv1 = pyg_nn.GATConv(in_channels=in_c, out_channels=hid_c, dropout=0.6 ,heads=1, concat=False)
        self.bn1   = nn.BatchNorm1d(hid_c)
        self.conv2 = pyg_nn.GATConv(in_channels=hid_c, out_channels=out_c, dropout=0.6, heads=1, concat=False)
    def forward(self, data,edge_type):
        # data.x  data.edge_index
        x = data.x  # [N, C], C为特征的维度
        # x = self.dfr(x)
        edge_index = data.edge_index[edge_type]  # [2, E], E为边的数量
        x = F.dropout(x, p=0.6, training=self.training)
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D], N是节点数量，D是第一层输出的隐藏层的维度
        hid = self.bn1(hid)
        hid = F.leaky_relu(hid)
        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c], out_c就是定义的输出
        out = F.log_softmax(out, dim=1)  # [N, out_c],表示输出
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask].long())
        return out,loss
    
class Model(nn.Module):
    def __init__(self,data_geo_x_shape,num_muti_gat,num_muti_mlp,num_node_features,data_x_N):
        super(Model, self).__init__()
        self.mutiGAT = MutiGAT(num_muti_gat=num_muti_gat,num_node_features=num_node_features, hid_c=16, out_c=2,data_x_N=data_x_N)
        # self.vdMLP = VariableDropoutMLP(data_x_shape=data_geo_x_shape)
        
        self.num_muti_mlp = num_muti_mlp
        self.vdMLP_list = []
        for i in range(num_muti_mlp):
            self.vdMLP_list.append(VariableDropoutMLP(data_x_shape=data_geo_x_shape[1]))
        self.vdMLP_list = nn.ModuleList(self.vdMLP_list)
    def forward(self,data,data_geo_x):
        vimp,loss_mutiGAT,out_graph,graph,pw_w = self.mutiGAT(data)
        out = torch.zeros([data_geo_x.shape[0],2]).to(next(self.parameters()).device)
        for module in self.vdMLP_list:
            temp = module(data_geo_x,out_graph)
            out = out + temp.exp()

        loss_L1 = torch.mean(torch.pow(out_graph,2))
        return {'out':torch.log(out/self.num_muti_mlp),'vimp_g':out_graph,'loss_mutiGAT':loss_mutiGAT,'loss_L1':loss_L1,'graph':graph,'temp':temp,'pw_w':pw_w,'cor':vimp}








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
    
    auc, f1, ap ,_= get_metrics(prediction[data.test_mask], target[data.test_mask])
    cor,_,_,_ = get_metrics(cor[data.test_mask], target[data.test_mask])
    auc_geo,f1_geo,ap_geo,acc_geo = get_metrics(out,data_geo.Y_test)
    auc_temp,_,_ ,_= get_metrics(temp,data_geo.Y_test)
    auc_geo_train,_,_,_ = get_metrics(out_train,data_geo.Y_train)
    

    model.train()
    # return auc, f1, ap, auc_geo,auc_geo_train,auc_temp,f1_geo,ap_geo
    return {'auc':auc,'f1':f1,'ap':ap,'auc_geo':auc_geo,'auc_geo_train':auc_geo_train,'auc_temp':auc_temp,'f1_geo':f1_geo,'ap_geo':ap_geo,'acc_geo':acc_geo,'cor':cor}






def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    df_acc = pd.DataFrame(columns=('epoch','auc_geo','auc_train','auc','loss'))
    
    data_geo_obj = make_data_geo(data_geo, label_geo,10,4,4709)
    

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
    for epoch in range(1000):
        optimizer.zero_grad()

        # np.random.seed(seed+i)
        lam = np.random.beta(alpha, alpha)
        # torch.manual_seed(seed+i)
        index = torch.randperm(data_geo_obj.X_train.size(0)).cuda()
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

    
    # model test
    
    my_net.eval()

    torch.save(my_net,"result/model.pt")

    result = my_net(data,data_geo_obj.X_test)
    pd.DataFrame({"predict":result['cor'].detach().cpu()}).to_csv("result/predict_muti_all.csv",index=False)
    pd.DataFrame({"predict":result['out'].max(dim=1).indices.detach().cpu()}).to_csv("result/predict_out.csv",index=False)
    pd.DataFrame(result['graph'].detach().cpu().numpy()).to_csv("result/graph.csv")
    pd.DataFrame({"predict":result['pw_w'].detach().cpu()}).to_csv("result/pw_w.csv",index=False)
    df_acc.to_csv("result/lossAndAcc.csv")

    test_= test(my_net, data,data_geo_obj)
    return test_




if __name__ == "__main__":
    main()
    print("finished")