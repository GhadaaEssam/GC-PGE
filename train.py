import os
from unittest import result
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
from model.preprocess import make_data_geo, get_train_edge, make_data, pgb, load_multiomics, make_data_multiomics
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

def test(model, data, data_geo):
    model.eval()
    target = data.y
    
    # 1. NEW: Pass all 4 Testing sets into the model
    result = model(data, data_geo.X_test_rna, x_meth=data_geo.X_test_meth, x_cnv=data_geo.X_test_cnv, x_snv=data_geo.X_test_snv)
    
    # 2. Extract the RAW probabilities (Required for calculating AUC correctly!)
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
    
def train_model(data_geo, label_geo, anchor_list, data_x, data_ppi_link_index, data_homolog_index,progressBarObj):
    
    if os.path.exists('result/'):
        pass
    else:
        os.mkdir('result/')
        os.mkdir('result/model/')

    # --- NEW MULTI-OMICS LOADING ---
    # 1. Load the labels directly from your new data folder
    label_df = pd.read_csv(r"data/Patients_labels.csv", header=0, index_col=0)
    label_geo = label_df.iloc[:, 0] # Grab the actual label values
    
    # 2. Peek at the RNA file to see which patients actually have RNA data
    rna_df = pd.read_csv(r"data/rna_ml.csv", header=0, index_col=0)
    
    # 3. Keep ONLY the patients that exist in BOTH the labels and the RNA file.
    valid_patients = label_geo.index.intersection(rna_df.index)
    label_geo = label_geo.loc[valid_patients]
    print(f"Found {len(valid_patients)} perfectly aligned patients. Loading Omics...")
    
    # 4. Load all 4 omics files using the clean, safe list of patients
    omics_dict = load_multiomics(
        data_geo=r"data/rna_ml.csv",
        data_meth=r"data/meth_ml.csv",
        data_cnv=r"data/cnv_ml.csv",
        data_snv=r"data/snv_ml.csv",
        sample_ids=valid_patients  # Pass the safe list here!
    )
    
    # 3. Create the multi-omics data object (this replaces make_data_geo)
    data_geo_obj = make_data_multiomics(omics_dict, label_geo, k=10, i=4, seed=4709)
    # -------------------------------

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor,test_anchor = model_selection.train_test_split(anchor_index, test_size=0.5)
    test_anchor_csv=pd.DataFrame(test_anchor)
    test_anchor_csv.to_csv(r'result/test_anchor.csv')

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor= pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))
    # --- THE BUG FIX ---
    # Translate the string gene names into numerical Node IDs (0, 1, 2...)
    # so they can successfully match with the numbers in the PPI network!
    train_anchor_numerical = pd.Series([data_x.index.get_loc(gene) for gene in train_anchor if gene in data_x.index])
    pgb1 = pgb(progressBarObj,0,20)
    train_edge_ppi , _ = get_train_edge(data_ppi_link_index, train_anchor_numerical,pgb1)
    
    pgb2 = pgb(progressBarObj,20,40)
    train_edge_homolog , _ = get_train_edge(data_homolog_index, train_anchor_numerical,pgb2)
    
    data_obj = make_data(data_x,train_edge_ppi,train_edge_homolog,anchor_list,test_anchor)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
    
    df_acc = pd.DataFrame(columns=('epoch','auc_geo','auc_train','auc','loss'))
    
    my_net = Model(data_geo_x_shape=data_geo_obj.X_train_rna.shape, num_muti_gat=8, num_muti_mlp=5, num_node_features=data_obj.num_node_features, data_x_N=data_obj.train_mask.shape[0])    # my_net = GraphCNN(in_c=data_obj.num_node_features, hid_c=8, out_c=2)
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

        # --- 1. THE MIXUP AUGMENTATION ---
        lam = np.random.beta(alpha, alpha)
        # We get ONE random index to shuffle all patients the exact same way
        index = torch.randperm(data_geo_obj.X_train_rna.size(0)).cpu()
        torch.seed()
        
        # We blend all 4 modalities using the exact same lam and index
        mixed_rna  = lam * data_geo_obj.X_train_rna  + (1 - lam) * data_geo_obj.X_train_rna[index, :]
        mixed_meth = lam * data_geo_obj.X_train_meth + (1 - lam) * data_geo_obj.X_train_meth[index, :]
        mixed_cnv  = lam * data_geo_obj.X_train_cnv  + (1 - lam) * data_geo_obj.X_train_cnv[index, :]
        mixed_snv  = lam * data_geo_obj.X_train_snv  + (1 - lam) * data_geo_obj.X_train_snv[index, :]

        # --- 2. THE FORWARD PASS ---
        # Pass all 4 mixed datasets into your friend's new engine
        result = my_net(data, mixed_rna, x_meth=mixed_meth, x_cnv=mixed_cnv, x_snv=mixed_snv)  
        
        out = result['out'] # The old RNA-only prediction
        out_multi = result['out_multiomics'] # The NEW Fused prediction
        
        loss_mutiGAT = result['loss_mutiGAT']
        loss_L1 = result['loss_L1']
        pw_w = result['pw_w']
        
        
        # --- 3. THE LOSS CALCULATION ---
        # Grade the old RNA path (Weight: 0.3)
        loss_rna = lam * F.nll_loss(out, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out, data_geo_obj.Y_train[index].long())
        
        # Grade the new Multi-Omics path (Weight: 1.0 - This is the primary goal!)
        loss_multi = lam * F.nll_loss(out_multi, data_geo_obj.Y_train.long()) + (1 - lam) * F.nll_loss(out_multi, data_geo_obj.Y_train[index].long())

        # Combine all the punishments together
        loss = 0.1 * loss_mutiGAT + 0.1 * torch.mean(torch.pow(pw_w,2)) + 0.1 * loss_L1 + 0.3 * loss_rna + 1.0 * loss_multi

        loss.backward()
        optimizer.step()  # Optimizer updates the weights!

        
        
        # out,_,loss_mutiGAT,loss_L1,_ = my_net(data,data_geo_obj.X_train)  # 预测结果
        # loss =  0.1*loss_mutiGAT +0.1*loss_L1 + 1.0*F.nll_loss(out, data_geo_obj.Y_train.long())
        
        # scheduler.step()
        test_= test(my_net, data,data_geo_obj)
        print("epoch:{},auc_geo:{},auc_train:{},auc:{},cor:{},ap:{},loss:{},auc_temp:{},num:{}".format(epoch + 1,test_['auc_geo'],test_['auc_geo_train'], test_['auc'], test_['cor'], test_['ap'] , loss.item(),test_['auc_temp'],num))
        # Create a tiny 1-row table for the current epoch's scores
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'auc_geo': [test_['auc_geo']],
            'auc_train': [test_['auc_geo_train']],
            'f1_geo': [test_['f1_geo']],
            'ap_geo': [test_['ap_geo']],
            'auc': [test_['auc']],
            'cor': [test_['cor']],
            'loss': [loss.item()],
            'auc_temp': [test_['auc_temp']],
            'acc_geo': [test_['acc_geo']]
        })
        
        # Use pd.concat to safely glue the new row to the main spreadsheet
        df_acc = pd.concat([df_acc, new_row], ignore_index=True)
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

    # Save as a NEW file name
    torch.save(my_net,"result/model_multiomics.pt")
    
    # Pass all 4 test datasets for the final CSV export
    result = my_net(data, data_geo_obj.X_test_rna, x_meth=data_geo_obj.X_test_meth, x_cnv=data_geo_obj.X_test_cnv, x_snv=data_geo_obj.X_test_snv)
    
    pd.DataFrame({"predict":result['cor'].detach().cpu()}).to_csv("result/predict_muti_all.csv",index=False)
    
    # Save the MULTI-OMICS prediction, not the old RNA one
    pd.DataFrame({"predict":result['out_multiomics'].max(dim=1).indices.detach().cpu()}).to_csv("result/predict_out.csv",index=False)
    
    pd.DataFrame(result['graph'].detach().cpu().numpy()).to_csv("result/graph.csv")
    pd.DataFrame({"predict":result['pw_w'].detach().cpu()}).to_csv("result/pw_w.csv",index=False)
    df_acc.to_csv("result/lossAndAcc.csv")

    progressBarObj.setValue(int(100))
    
# =====================================================================
# STANDALONE EXECUTION BLOCK
# This tells Python to actually run the code when you type 'python train.py'
# =====================================================================
if __name__ == "__main__":
    print("🚀 Booting up the Multi-Omics Training Engine...")
    
    # 1. Create a fake progress bar so the GUI code doesn't crash
    class DummySignal:
        def emit(self, val):
            if val % 10 == 0:
                print(f"⏳ Graph Processing: {val}%")
        def setValue(self, val):
            print(f"✅ Process Complete: {val}%")
    
    dummy_pgb = DummySignal()
    
    # 2. Load the massive biological networks into memory
    print("📂 Loading massive biological networks (this might take a moment)...")
    anchor_list = pd.read_csv(r'data/BRCA_pubmed_results.csv', header=0, index_col=0)
    data_x = pd.read_csv(r'data/BRCA_data_x_all.csv', header=0, index_col=0)
    data_ppi = pd.read_csv(r'data/ppi_final_edge_list.csv', header=0)
    data_homolog = pd.read_csv(r'data/homology_final_edge_list.csv', header=0)
    
    # 3. Pull the trigger! 
    # (We pass 'None' for the first two arguments because you already wrote 
    # the code to load the Omics and Labels directly inside the function!)
    print("🧠 Initializing Deep Learning sequence...")
    train_model(None, None, anchor_list, data_x, data_ppi, data_homolog, dummy_pgb)