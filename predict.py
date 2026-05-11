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

# --- NEW: Now loads all 4 Multi-Omics layers ---
def make_data_multiomics_no_label(omics_dict):
    data = Data()
    data.X_rna = torch.tensor(omics_dict['data_geo_x'].values, dtype=torch.float)
    data.X_meth = torch.tensor(omics_dict['data_meth_x'].values, dtype=torch.float)
    data.X_cnv = torch.tensor(omics_dict['data_cnv_x'].values, dtype=torch.float)
    data.X_snv = torch.tensor(omics_dict['data_snv_x'].values, dtype=torch.float)
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


# --- EDITED: Changed data_geo to omics_dict to match train.py ---
def predict_model_multiomics(model_path, omics_dict, anchor_list, data_x, data_ppi_link_index, data_homolog_index, progressBarObj):
    print("🧬 Initializing 5-Model Ensemble Pipeline...")
    
    data_geo_obj = make_data_multiomics_no_label(omics_dict)

    anchor_index = anchor_list.result_num[anchor_list.result_num==1].index
    train_anchor, test_anchor = model_selection.train_test_split(anchor_index, test_size=0.2)
    
    test_anchor_csv = pd.DataFrame(test_anchor)
    test_anchor_csv.to_csv(r'result/test_anchor.csv')

    train_anchor = pd.Series(list(set(anchor_index.to_list())-set(test_anchor.to_list())))
    train_anchor_numerical = pd.Series([data_x.index.get_loc(gene) for gene in train_anchor if gene in data_x.index])

    print("📂 Loading PPI network...")
    pgb1 = pgb(progressBarObj, 0, 30)
    train_edge_ppi, _ = get_train_edge(data_ppi_link_index, train_anchor_numerical, pgb1)
    
    print("📂 Loading Homolog network...")
    pgb2 = pgb(progressBarObj, 30, 60)
    train_edge_homolog, _ = get_train_edge(data_homolog_index, train_anchor_numerical, pgb2)
    
    data_obj = make_data(data_x, train_edge_ppi, train_edge_homolog, anchor_list, test_anchor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = data_obj.to(device)
    data_geo_obj = data_geo_obj.to(device)
    
    # --- ENSEMBLE SOFT VOTING SYSTEM ---
    num_patients = data_geo_obj.X_rna.shape[0]
    
    # Create empty containers to add up all the probabilities
    sum_out_multi = torch.zeros((num_patients, 2)).to(device)
    sum_cor = 0
    sum_graph = 0
    sum_pw_w = 0
    
    print("\n🩺 Assembling Medical Board (Loading 5 Models)...")
    
    # Loop through all 5 trained folds
    for i in range(1, 6):
        current_model_path = f"result/model_multiomics_Fold_{i}.pt"
        
        if not os.path.exists(current_model_path):
            print(f"⚠️ WARNING: {current_model_path} not found. Ensure all 5 folds are trained!")
            continue
            
        print(f"   - Consulting Model Fold {i}...")
        
        # Load the model using the security bypass fix!
        my_net = torch.load(current_model_path, map_location=device, weights_only=False)
        my_net.eval()
        
        with torch.no_grad():
            result = my_net(data, data_geo_obj.X_rna, x_meth=data_geo_obj.X_meth, x_cnv=data_geo_obj.X_cnv, x_snv=data_geo_obj.X_snv)  
            
            # Add the raw numbers from this model to our running totals
            sum_out_multi += torch.exp(result['out_multiomics']) # Reverse LogSoftmax to get true probability
            sum_cor += result['cor']
            sum_graph += result['graph']
            sum_pw_w += result['pw_w']
            
        # Update progress bar for each model loaded
        if hasattr(progressBarObj, 'setValue'):
            progressBarObj.setValue(60 + (i * 8)) # Scales from 60% to 100%

    # --- CALCULATE THE FINAL AVERAGES ---
    # Divide the totals by 5 to get the exact Mean Probability across all models
    avg_out_multi = sum_out_multi / 5.0
    avg_cor = sum_cor / 5.0
    avg_graph = sum_graph / 5.0
    avg_pw_w = sum_pw_w / 5.0
    
    # --- SAVE THE ENSEMBLE RESULTS ---
    pd.DataFrame({"predict": avg_cor.detach().cpu()}).to_csv("result/predict_muti_all.csv", index=False)
    
    # The final diagnosis is the class (0 or 1) with the highest average probability!
    final_diagnosis = avg_out_multi.argmax(dim=1).detach().cpu()
    pd.DataFrame({"predict": final_diagnosis}).to_csv("result/predict_out.csv", index=False)
    
    pd.DataFrame(avg_graph.detach().cpu().numpy()).to_csv("result/graph.csv")
    pd.DataFrame({"predict": avg_pw_w.detach().cpu()}).to_csv("result/pw_w.csv", index=False)

    print("\n✅ Ensemble Prediction Complete! Results saved to CSV.")
    if hasattr(progressBarObj, 'setValue'):
        progressBarObj.setValue(int(100))

# =====================================================================
# STANDALONE TESTING BLOCK
# =====================================================================
if __name__ == "__main__":
    from model.preprocess import load_multiomics
    print("🚀 Booting up the Prediction Engine Test...")
    
    class DummySignal:
        def emit(self, val): pass
        def setValue(self, val): print(f"✅ Progress Complete: {val}%")
    
    dummy_pgb = DummySignal()
    
    print("📂 Loading biological networks...")
    anchor_list = pd.read_csv(r'data/BRCA_pubmed_results.csv', header=0, index_col=0)
    data_x = pd.read_csv(r'data/BRCA_data_x_all.csv', header=0, index_col=0)
    data_ppi = pd.read_csv(r'data/ppi_final_edge_list.csv', header=0)
    data_homolog = pd.read_csv(r'data/homology_final_edge_list.csv', header=0)
    
    omics_dict = load_multiomics(
        data_geo=r"data/rna_ml.csv",
        data_meth=r"data/meth_ml.csv",
        data_cnv=r"data/cnv_ml.csv",
        data_snv=r"data/snv_ml.csv",
        sample_ids=pd.read_csv(r"data/rna_ml.csv", header=0, index_col=0).index.tolist()
    )
    
    # Dummy string passed just to fulfill the argument, the script uses the result folder automatically!
    predict_model_multiomics("ignore_me", omics_dict, anchor_list, data_x, data_ppi, data_homolog, dummy_pgb)