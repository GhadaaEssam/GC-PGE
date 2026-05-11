import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import numpy as np


EPSILON = np.finfo(float).eps

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