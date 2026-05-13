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
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn import model_selection
import numpy as np
from model.preprocess import make_data_geo, get_train_edge, make_data, pgb, load_multiomics, make_data_multiomics
from scipy.special import erfinv 
from model.model import Model


EPSILON = np.finfo(float).eps

# =====================================================================
# SHARED METRICS GRADER
# =====================================================================
def get_metrics(out_, edge_label_):
    out = out_.detach().cpu().numpy()
    edge_label = edge_label_.detach().cpu().numpy()
    
    if len(np.unique(edge_label)) < 2:
        return 0.5, 0.0, 0.0, 0.0
    
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred, zero_division=0)
    accuracy = accuracy_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap, accuracy

# =====================================================================
# 1. SINGLE-OMICS PIPELINE (THE OLD BASE MODEL)
# =====================================================================
def test(model, data, data_geo):
    model.eval()
    target = data.y
    # out,prediction,_ ,_,_,temp= model(data,data_geo.X_test)
    result= model(data,data_geo.X_test)
    out = result['out']
    prediction = result['vimp_g']
    temp = result['temp']
    cor = result['cor']

    out = out.max(dim=1).indices
    temp = temp.max(dim=1).indices

    result_ = model(data, data_geo.X_train)
    out_train = result_['out'].max(dim=1).indices
    
    auc1, f1, ap, _ = get_metrics(prediction[data.test_mask], target[data.test_mask])
    auc, _, _, _ = get_metrics(cor[data.test_mask], target[data.test_mask])
    auc_geo, f1_geo, ap_geo, acc_geo = get_metrics(out, data_geo.Y_test)
    auc_temp, _, _, _ = get_metrics(temp, data_geo.Y_test)
    auc_geo_train, _, _, _ = get_metrics(out_train, data_geo.Y_train)
    
    model.train()
    # return auc, f1, ap, auc_geo,auc_geo_train,auc_temp,f1_geo,ap_geo
    return {'auc':auc,'f1':f1,'ap':ap,'auc_geo':auc_geo,'auc_geo_train':auc_geo_train,'auc_temp':auc_temp,'f1_geo':f1_geo,'ap_geo':ap_geo,'acc_geo':acc_geo,'cor':auc1}

def train_model(data_geo, label_geo, anchor_list, data_x, data_ppi_link_index, data_homolog_index,progressBarObj):
    
    if os.path.exists('result/'):
        pass
    else:
        os.mkdir('result/')
        os.mkdir('result/model/')
        
    if isinstance(label_geo, pd.Series) or isinstance(label_geo, pd.DataFrame):
        label_geo = label_geo.reset_index(drop=True)
    
    # RankGauss Normalization (Old Method)
    rankGauss = (data_geo.values / data_geo.values.max() - 0.5) * 2
    rankGauss = np.clip(rankGauss, -1 + EPSILON, 1 - EPSILON)
    rankGauss = erfinv(rankGauss) 
    data_geo = pd.DataFrame(rankGauss, columns=data_geo.columns)
    data_geo_obj = make_data_geo(data_geo, label_geo, 10, 4, 4709)

    anchor_index = anchor_list.result_num[anchor_list.result_num == 1].index
    train_anchor, test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
    
    test_anchor_csv = pd.DataFrame(test_anchor)
    test_anchor_csv.to_csv(r'result/test_anchor.csv')

    train_anchor = pd.Series(list(set(anchor_index.to_list()) - set(test_anchor.to_list())))

    pgb1 = pgb(progressBarObj, 0, 20)
    train_edge_ppi, _ = get_train_edge(data_ppi_link_index, train_anchor, pgb1)
    
    pgb2 = pgb(progressBarObj, 20, 40)
    train_edge_homolog, _ = get_train_edge(data_homolog_index, train_anchor, pgb2)
    
    data_obj = make_data(data_x, train_edge_ppi, train_edge_homolog, anchor_list, test_anchor)

    df_acc = pd.DataFrame(columns=('epoch', 'auc_geo', 'auc_train', 'auc', 'loss'))
    
    
    

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
    pgb3 = pgb(progressBarObj, 40, 98)

    for epoch in range(epoches):
        optimizer.zero_grad()

        # np.random.seed(seed+i)
        lam = np.random.beta(alpha, alpha)
        # torch.manual_seed(seed+i)
        index = torch.randperm(data_geo_obj.X_train.size(0)).cpu()
        torch.seed()
        mixed_x = lam * data_geo_obj.X_train + (1 - lam) * data_geo_obj.X_train[index, :]

        result = my_net(data, mixed_x) 
        out = result['out']
        loss_mutiGAT = result['loss_mutiGAT']
        loss_L1 = result['loss_L1']
        pw_w = result['pw_w']
        loss =   0.1*loss_mutiGAT +0.1*torch.mean(torch.pow(pw_w,2))+0.1*loss_L1 + 1.0*(lam * F.nll_loss(out, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out, data_geo_obj.Y_train[index].long()))


        # out,_,loss_mutiGAT,loss_L1,_ = my_net(data,data_geo_obj.X_train)  # 预测结果
        # loss =  0.1*loss_mutiGAT +0.1*loss_L1 + 1.0*F.nll_loss(out, data_geo_obj.Y_train.long())


        loss.backward()
        optimizer.step()  # 优化器
        # scheduler.step()
        test_= test(my_net, data,data_geo_obj)
        print("epoch:{},auc_geo:{},auc_train:{},auc:{},cor:{},ap:{},loss:{},auc_temp:{},num:{}".format(epoch + 1,test_['auc_geo'],test_['auc_geo_train'], test_['auc'], test_['cor'], test_['ap'] , loss.item(),test_['auc_temp'],num))
        df_acc=df_acc._append(pd.DataFrame({'epoch':[epoch],'auc_geo':[test_['auc_geo']],'auc_train':[test_['auc_geo_train']],'f1_geo':[test_['f1_geo']],'ap_geo':test_['ap_geo'],'auc':[test_['auc']],'cor':[test_['cor']],'loss':[loss.item()],'auc_temp':[test_['auc_temp']],'acc_geo':[test_['acc_geo']]}),ignore_index=True)
        pgb3.update((epoch+1)/epoches)


        
        # if  test_['auc_geo_train'] > 0.99 and epoch>=235: #and epoch>=250 
        #     num = num - 1
        # else:
        #     num = 10
        # if (num == 0 and auc_stock == test_['auc_geo_train']):
        #     print("###")
        #     break    
        
        if (auc_stock <= test_['auc_geo_train']) and test_['auc_geo_train'] > 0.99:
            num = num - 1
            auc_stock = test_['auc_geo_train']
        else:
            num = 5
            auc_stock = test_['auc_geo_train']
        if (num == 0 and auc_stock <= test_['auc_geo_train']):
            print("### Early Stopping ###")
            break
        

    my_net.eval()
    
    torch.save(my_net, "result/model.pt")
    result = my_net(data, data_geo_obj.X_test)
    pd.DataFrame({"predict": result['cor'].detach().cpu()}).to_csv("result/predict_muti_all.csv", index=False)
    pd.DataFrame({"predict": result['out'].max(dim=1).indices.detach().cpu()}).to_csv("result/predict_out.csv", index=False)
    pd.DataFrame(result['graph'].detach().cpu().numpy()).to_csv("result/graph.csv")
    pd.DataFrame({"predict": result['pw_w'].detach().cpu()}).to_csv("result/pw_w.csv", index=False)
    df_acc.to_csv("result/lossAndAcc.csv")
    
    if hasattr(progressBarObj, 'setValue'): progressBarObj.setValue(int(100))


# =====================================================================
# 2. MULTI-OMICS PIPELINE (THE NEW FUSED ENGINE)
# =====================================================================
def test_multi(model, data, data_geo):
    model.eval()
    target = data.y
    
    # 1. NEW: Pass all 4 Testing sets into the model
    result = model(data, data_geo.X_test_rna, x_meth=data_geo.X_test_meth, x_cnv=data_geo.X_test_cnv, x_snv=data_geo.X_test_snv)
    
    # 2. Extract the RAW probabilities (Restored to exact high-accuracy math)
    out = result['out_multiomics'][:, 1] 
    prediction = result['vimp_g']
    temp = result['temp'][:, 1]
    cor = result['cor']
    
    # 3. NEW: Pass all 4 Training sets for the baseline check
    result_ = model(data, data_geo.X_train_rna, x_meth=data_geo.X_train_meth, x_cnv=data_geo.X_train_cnv, x_snv=data_geo.X_train_snv)
    out_train = result_['out_multiomics'][:, 1]
    
    # 4. Calculate the metrics using the multi-omics outputs
    auc1, f1, ap, _ = get_metrics(prediction[data.test_mask], target[data.test_mask])
    auc, _, _, _ = get_metrics(cor[data.test_mask], target[data.test_mask])
    
    auc_geo, f1_geo, ap_geo, acc_geo = get_metrics(out, data_geo.Y_test.long())
    auc_temp, _, _, _ = get_metrics(temp, data_geo.Y_test.long())
    auc_geo_train, _, _, _ = get_metrics(out_train, data_geo.Y_train.long())
    
    model.train() 
    
    return {
        'auc': auc, 'f1': f1, 'ap': ap, 
        'auc_geo': auc_geo, 'auc_geo_train': auc_geo_train, 
        'auc_temp': auc_temp, 'f1_geo': f1_geo, 'ap_geo': ap_geo, 
        'acc_geo': acc_geo, 'cor': auc1
    }

def train_model_multi(data_geo, label_geo, anchor_list, data_x, data_ppi_link_index, data_homolog_index, progressBarObj, current_fold=0):
    # if not os.path.exists('result/'):
    #     pass
    # else:
    #     os.mkdir('result/')
    #     os.mkdir('result/model/')
    
    # Safely creates the folders if missing, safely ignores them if they exist!
    os.makedirs('result/', exist_ok=True)

    # --- NEW MULTI-OMICS LOADING ---
    label_df = pd.read_csv(r"data/Patients_labels.csv", header=0, index_col=0)
    label_geo = label_df.iloc[:, 0] 
    
    rna_df = pd.read_csv(r"data/rna_ml.csv", header=0, index_col=0)
    
    valid_patients = label_geo.index.intersection(rna_df.index)
    label_geo = label_geo.loc[valid_patients]
    print(f"Found {len(valid_patients)} perfectly aligned patients. Loading Omics...")
    
    omics_dict = load_multiomics(
        data_geo=r"data/rna_ml.csv",
        data_meth=r"data/meth_ml.csv",
        data_cnv=r"data/cnv_ml.csv",
        data_snv=r"data/snv_ml.csv",
        sample_ids=valid_patients 
    )
    
    data_geo_obj = make_data_multiomics(omics_dict, label_geo, k=5, i=current_fold, seed=4709)

    # RESTORED: The original 50% anchor split that works best for your dataset
    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor, test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
    
    test_anchor_csv=pd.DataFrame(test_anchor)
    test_anchor_csv.to_csv(f'result/test_anchor_Fold_{current_fold+1}.csv')

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))
    
    train_anchor_numerical = pd.Series([data_x.index.get_loc(gene) for gene in train_anchor if gene in data_x.index])
    
    pgb1 = pgb(progressBarObj, 0, 20)
    train_edge_ppi , _ = get_train_edge(data_ppi_link_index, train_anchor_numerical, pgb1)
    pgb2 = pgb(progressBarObj, 20, 40)
    train_edge_homolog , _ = get_train_edge(data_homolog_index, train_anchor_numerical, pgb2)
    
    data_obj = make_data(data_x, train_edge_ppi, train_edge_homolog, anchor_list, test_anchor)

    df_acc = pd.DataFrame(columns=('epoch','auc_geo','auc_train','auc','loss'))
    
    # RESTORED: The original high-performing Model Size (8 and 5)
    my_net = Model(data_geo_x_shape=data_geo_obj.X_train_rna.shape, num_muti_gat=8, num_muti_mlp=5, num_node_features=data_obj.num_node_features, data_x_N=data_obj.train_mask.shape[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_net = my_net.to(device)  
    data = data_obj.to(device)  
    data_geo_obj = data_geo_obj.to(device)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.0005, weight_decay=1e-2) 
    
    alpha = 0.5
    auc_stock = 0.0
    num = 10
    my_net.train()
    epoches = 500
    pgb3 = pgb(progressBarObj,40,98)

    for epoch in range(epoches):
        optimizer.zero_grad()

        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(data_geo_obj.X_train_rna.size(0)).cpu()
        torch.seed()
        
        mixed_rna  = lam * data_geo_obj.X_train_rna  + (1 - lam) * data_geo_obj.X_train_rna[index, :]
        mixed_meth = lam * data_geo_obj.X_train_meth + (1 - lam) * data_geo_obj.X_train_meth[index, :]
        mixed_cnv  = lam * data_geo_obj.X_train_cnv  + (1 - lam) * data_geo_obj.X_train_cnv[index, :]
        mixed_snv  = lam * data_geo_obj.X_train_snv  + (1 - lam) * data_geo_obj.X_train_snv[index, :]

        result = my_net(data, mixed_rna, x_meth=mixed_meth, x_cnv=mixed_cnv, x_snv=mixed_snv)  
        
        out = result['out'] 
        out_multi = result['out_multiomics']
        loss_mutiGAT = result['loss_mutiGAT']
        loss_L1 = result['loss_L1']
        pw_w = result['pw_w']
        
        loss_rna = lam * F.nll_loss(out, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out, data_geo_obj.Y_train[index].long())
        loss_multi = lam * F.nll_loss(out_multi, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out_multi, data_geo_obj.Y_train[index].long())

        # RESTORED: The exact original loss balance
        loss = 0.1 * loss_mutiGAT + 0.1 * torch.mean(torch.pow(pw_w,2)) + 0.1 * loss_L1 + 0.3 * loss_rna + 1.0 * loss_multi

        loss.backward()
        optimizer.step()

        test_= test_multi(my_net, data, data_geo_obj)
        print("epoch:{},auc_geo:{},auc_train:{},auc:{},cor:{},ap:{},loss:{},auc_temp:{},num:{}".format(epoch + 1,test_['auc_geo'],test_['auc_geo_train'], test_['auc'], test_['cor'], test_['ap'] , loss.item(),test_['auc_temp'],num))
        
        new_row = pd.DataFrame({
            'epoch': [epoch], 'auc_geo': [test_['auc_geo']], 'auc_train': [test_['auc_geo_train']],
            'f1_geo': [test_['f1_geo']], 'ap_geo': [test_['ap_geo']], 'auc': [test_['auc']],
            'cor': [test_['cor']], 'loss': [loss.item()], 'auc_temp': [test_['auc_temp']],
            'acc_geo': [test_['acc_geo']]
        })
        
        df_acc = pd.concat([df_acc, new_row], ignore_index=True)
        pgb3.update((epoch+1)/epoches)
        
        if (auc_stock <= test_['auc_geo_train']) and test_['auc_geo_train'] > 0.99 : 
            num = num - 1
            auc_stock = test_['auc_geo_train']
        else:
            num = 5
            auc_stock = test_['auc_geo_train']
        if (num == 0 and auc_stock <= test_['auc_geo_train']):
            print("### Early Stopping ###")
            break

    my_net.eval()

    torch.save(my_net, f"result/model_multiomics_Fold_{current_fold+1}.pt")
    
    result = my_net(data, data_geo_obj.X_test_rna, x_meth=data_geo_obj.X_test_meth, x_cnv=data_geo_obj.X_test_cnv, x_snv=data_geo_obj.X_test_snv)
    pd.DataFrame({"predict":result['cor'].detach().cpu()}).to_csv(f"result/predict_muti_all_Fold_{current_fold+1}.csv",index=False)
    pd.DataFrame({"predict":result['out_multiomics'].max(dim=1).indices.detach().cpu()}).to_csv(f"result/predict_out_Fold_{current_fold+1}.csv",index=False)
    pd.DataFrame(result['graph'].detach().cpu().numpy()).to_csv(f"result/graph_Fold_{current_fold+1}.csv")
    pd.DataFrame({"predict":result['pw_w'].detach().cpu()}).to_csv(f"result/pw_w_Fold_{current_fold+1}.csv",index=False)
    df_acc.to_csv(f"result/lossAndAcc_Fold_{current_fold+1}.csv")

    if hasattr(progressBarObj, 'setValue'):
        progressBarObj.setValue(int(100))
    
# =====================================================================
# THE MASTER SWITCH
# =====================================================================
if __name__ == "__main__":
    # --- CHOOSE YOUR ENGINE HERE ---
    RUN_MODE = "SINGLE" 
    # -------------------------------
    
    print("🚀 Booting up the Training Engine...")
    
    class DummySignal:
        def emit(self, val):
            if val % 10 == 0: print(f"⏳ Graph Processing: {val}%")
        def setValue(self, val):
            print(f"✅ Process Complete: {val}%")
    
    dummy_pgb = DummySignal()
    
    print("📂 Loading massive biological networks...")
    anchor_list = pd.read_csv(r'data/BRCA_pubmed_results.csv', header=0, index_col=0)
    data_x = pd.read_csv(r'data/BRCA_data_x_all.csv', header=0, index_col=0)
    data_ppi = pd.read_csv(r'data/ppi_final_edge_list.csv', header=0)
    data_homolog = pd.read_csv(r'data/homology_final_edge_list.csv', header=0)

    if RUN_MODE == "SINGLE":
        print("\n🚀 Starting the SINGLE-OMICS Engine...")
        anchor_list = pd.read_csv(r'data/2-pubmed_result.csv', header=0, index_col=0)
        data_x = pd.read_csv(r'data/data_x_all.csv', header=0, index_col=0)
        data_ppi = pd.read_csv(r'data/ppi_link_Breast Cancer TFAC_600.csv', header=0)
        data_homolog = pd.read_csv(r'data/homolog_Breast Cancer TFAC.csv', header=0)
        label_df = pd.read_csv(r"data/sample.csv", header=0, index_col=0)
        rna_df = pd.read_csv(r"data/1-Breast Cancer TFAC_result_test_rank.csv", header=0, index_col=0)
        
        valid_patients = label_df.iloc[:, 0].index.intersection(rna_df.index)
        label_geo = label_df.iloc[:, 0].loc[valid_patients]
        data_geo = rna_df.loc[valid_patients]
        
        train_model(data_geo, label_geo, anchor_list, data_x, data_ppi, data_homolog, dummy_pgb)
        
    elif RUN_MODE == "MULTI":
        print("\n🚀 Starting the MULTI-OMICS 5-Fold Engine...")
        for current_fold in range(5):
            print(f"\n=======================================================")
            print(f"🚀 STARTING FOLD {current_fold + 1} OF 5")
            print(f"=======================================================")
            train_model_multi(None, None, anchor_list, data_x, data_ppi, data_homolog, dummy_pgb, current_fold)
        print("\n✅ All 5 Folds Complete!")