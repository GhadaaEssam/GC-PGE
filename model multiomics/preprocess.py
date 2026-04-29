import torch
from torch_geometric.data import Data
import pandas as pd
import random
from sklearn import model_selection
import numpy as np

def _load_omics_frame(data_source, index_col=0):
    if isinstance(data_source, pd.DataFrame):
        frame = data_source.copy()
        if index_col is not None and isinstance(frame.index, pd.RangeIndex):
            if isinstance(index_col, int):
                if index_col < 0 or index_col >= frame.shape[1]:
                    raise IndexError(f"index_col={index_col} is out of bounds for frame with {frame.shape[1]} columns")
                frame = frame.set_index(frame.columns[index_col])
            else:
                if index_col not in frame.columns:
                    raise KeyError(f"index_col '{index_col}' not found in DataFrame columns")
                frame = frame.set_index(index_col)
        return frame

    if isinstance(data_source, str):
        return pd.read_csv(data_source, header=0, index_col=index_col)

    raise TypeError("Each omics input must be either a pandas DataFrame or a CSV file path")


def _coerce_numeric_frame(frame, name):
    numeric_frame = frame.apply(pd.to_numeric, errors='coerce')
    if numeric_frame.isnull().values.any():
        raise ValueError(f"{name} contains non-numeric values after removing the sample-id column")
    return numeric_frame


def _normalize_omics_axis(frame, name, genes_axis=1):
    if genes_axis not in (0, 1):
        raise ValueError("genes_axis must be 0 (genes in rows) or 1 (genes in columns)")

    normalized = frame.copy()
    if genes_axis == 0:
        normalized = normalized.T

    normalized.index = normalized.index.map(str)
    normalized.columns = normalized.columns.map(str)
    normalized = normalized.loc[~normalized.index.duplicated(keep='first')]
    normalized = normalized.loc[:, ~normalized.columns.duplicated(keep='first')]
    normalized = _coerce_numeric_frame(normalized, name)
    return normalized


def load_multiomics(
    data_geo,
    data_meth,
    data_cnv,
    data_snv,
    index_col=0,
    genes_axis=1,
    sample_ids=None,
    gene_ids=None,
):
    omics_frames = {
        'data_geo_x': _normalize_omics_axis(_load_omics_frame(data_geo, index_col=index_col), 'data_geo_x', genes_axis=genes_axis),
        'data_meth_x': _normalize_omics_axis(_load_omics_frame(data_meth, index_col=index_col), 'data_meth_x', genes_axis=genes_axis),
        'data_cnv_x': _normalize_omics_axis(_load_omics_frame(data_cnv, index_col=index_col), 'data_cnv_x', genes_axis=genes_axis),
        'data_snv_x': _normalize_omics_axis(_load_omics_frame(data_snv, index_col=index_col), 'data_snv_x', genes_axis=genes_axis),
    }

    sample_sets = [set(frame.index) for frame in omics_frames.values()]
    gene_sets = [set(frame.columns) for frame in omics_frames.values()]
    geo_frame = omics_frames['data_geo_x']

    if sample_ids is None:
        shared_samples = set.intersection(*sample_sets)
        aligned_samples = [sample_id for sample_id in geo_frame.index if sample_id in shared_samples]
    else:
        aligned_samples = [str(sample_id) for sample_id in sample_ids]

    if gene_ids is None:
        shared_genes = set.intersection(*gene_sets)
        aligned_genes = [gene_id for gene_id in geo_frame.columns if gene_id in shared_genes]
    else:
        aligned_genes = [str(gene_id) for gene_id in gene_ids]

    if len(aligned_samples) == 0:
        raise ValueError("No shared samples were found across the omics matrices")
    if len(aligned_genes) == 0:
        raise ValueError("No shared genes were found across the omics matrices")

    for name, frame in omics_frames.items():
        missing_samples = set(aligned_samples) - set(frame.index)
        missing_genes = set(aligned_genes) - set(frame.columns)
        if missing_samples:
            raise ValueError(f"{name} is missing {len(missing_samples)} aligned samples")
        if missing_genes:
            raise ValueError(f"{name} is missing {len(missing_genes)} aligned genes")
        omics_frames[name] = frame.loc[aligned_samples, aligned_genes]

    data_geo_x = omics_frames['data_geo_x']
    data_meth_x = omics_frames['data_meth_x']
    data_cnv_x = omics_frames['data_cnv_x']
    data_snv_x = omics_frames['data_snv_x']

    if not (
        data_geo_x.shape[1] == data_meth_x.shape[1]
        == data_cnv_x.shape[1] == data_snv_x.shape[1]
    ):
        raise ValueError("Aligned omics matrices do not share the same gene dimension")

    if not (
        data_geo_x.shape[0] == data_meth_x.shape[0]
        == data_cnv_x.shape[0] == data_snv_x.shape[0]
    ):
        raise ValueError("Aligned omics matrices do not share the same sample dimension")

    return {
        'data_geo_x': data_geo_x,
        'data_meth_x': data_meth_x,
        'data_cnv_x': data_cnv_x,
        'data_snv_x': data_snv_x,
        'sample_ids': data_geo_x.index.to_list(),
        'gene_ids': data_geo_x.columns.to_list(),
    }

def make_data_multiomics(omics_dict, labels, k, i, seed=42):
    assert k > 1

    data = Data()

    data_geo_x = omics_dict['data_geo_x']
    data_meth_x = omics_dict['data_meth_x']
    data_cnv_x = omics_dict['data_cnv_x']
    data_snv_x = omics_dict['data_snv_x']

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError("labels DataFrame must contain exactly one label column")
        label_series = labels.iloc[:, 0].copy()
    else:
        label_series = labels.copy()

    sample_ids = [str(sample_id) for sample_id in omics_dict['sample_ids']]
    label_series.index = label_series.index.map(str)

    missing_labels = set(sample_ids) - set(label_series.index)
    if missing_labels:
        raise ValueError(f"labels are missing {len(missing_labels)} aligned samples")

    label_series = label_series.loc[sample_ids]

    np.random.seed(seed)
    indices = np.random.permutation(range(len(label_series)))

    X_rna = torch.tensor(data_geo_x.values, dtype=torch.float)
    X_meth = torch.tensor(data_meth_x.values, dtype=torch.float)
    X_cnv = torch.tensor(data_cnv_x.values, dtype=torch.float)
    X_snv = torch.tensor(data_snv_x.values, dtype=torch.float)
    Y = torch.tensor(label_series.values, dtype=torch.int)

    fold_size = X_rna.shape[0] // k

    X_train_rna, X_test_rna = None, None
    X_train_meth, X_test_meth = None, None
    X_train_cnv, X_test_cnv = None, None
    X_train_snv, X_test_snv = None, None
    Y_train, Y_test = None, None

    for j in range(k):
        idx = indices[j * fold_size:(j + 1) * fold_size]

        X_part_rna = X_rna[idx, :]
        X_part_meth = X_meth[idx, :]
        X_part_cnv = X_cnv[idx, :]
        X_part_snv = X_snv[idx, :]
        y_part = Y[idx]

        if j == i:
            X_test_rna = X_part_rna
            X_test_meth = X_part_meth
            X_test_cnv = X_part_cnv
            X_test_snv = X_part_snv
            Y_test = y_part
        elif X_train_rna is None:
            X_train_rna = X_part_rna
            X_train_meth = X_part_meth
            X_train_cnv = X_part_cnv
            X_train_snv = X_part_snv
            Y_train = y_part
        else:
            X_train_rna = torch.cat((X_train_rna, X_part_rna), dim=0)
            X_train_meth = torch.cat((X_train_meth, X_part_meth), dim=0)
            X_train_cnv = torch.cat((X_train_cnv, X_part_cnv), dim=0)
            X_train_snv = torch.cat((X_train_snv, X_part_snv), dim=0)
            Y_train = torch.cat((Y_train, y_part), dim=0)

    data.X_train_rna = X_train_rna
    data.X_test_rna = X_test_rna
    data.X_train_meth = X_train_meth
    data.X_test_meth = X_test_meth
    data.X_train_cnv = X_train_cnv
    data.X_test_cnv = X_test_cnv
    data.X_train_snv = X_train_snv
    data.X_test_snv = X_test_snv
    data.Y_train = Y_train
    data.Y_test = Y_test

    return data

class pgb():
    def __init__(self,signalObj,min_value,max_value):
        self.min_value = min_value
        
        self.scale = max_value-min_value
        self.signalObj = signalObj
    def update(self,n_value):
        self.signalObj.emit(int(n_value*self.scale + self.min_value))
        


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

def get_train_edge(data_edge_index, train_anchor, pgb):
    train_edge_index = pd.DataFrame(dtype=int)
    test_edge_index = pd.DataFrame(dtype=int)

    for i in range(len(data_edge_index)):
        if(i%1000 == 0):
            # print(i/len(data_edge_index))
            pgb.update(i/len(data_edge_index))
        if (data_edge_index.iloc[i,0] in train_anchor.values) and (data_edge_index.iloc[i,1] in train_anchor.values):
            train_edge_index = train_edge_index.append(data_edge_index.iloc[i,:])
        elif(data_edge_index.iloc[i,0] in train_anchor.values) or (data_edge_index.iloc[i,1] in train_anchor.values):
            test_edge_index = test_edge_index.append(data_edge_index.iloc[i,:])
    return train_edge_index , test_edge_index

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
