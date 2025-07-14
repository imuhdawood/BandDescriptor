from ast import arg
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from model import *
import argparse
from pathlib import Path
from gutils import load_graphs

import sys
sys.path.append('.')


from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL, FEATS_DICT
from helper_functions import (
load_counts_matrix,
 select_within_polygons,
 create_geojson,
 load_and_preprocess_annot,
 combine_geojson,
 save_geojson,
 generate_cell_adjacency_graph,
 read_it_regions_annotations,
 mkdirs,
 read_geojson,
 transform_coordinates,
 transform_geometries,
 compute_surface_distances
 )



if __name__ == '__main__':
    
    FEAT = 'CELL_TYPES_BAND_DECRIPTOR' #'ENVIRON_TYPE' # 'CELL_TYPES'
    ENVIRON_COL = 'ENVIRON'
    CELL_COL = 'final_lineage'
    TAG = 'EXP_5'
    radii = '1'
    interaction_radius_microns = 50
    returnNodeProba = False
    ROOT_DIR = PROJECT_DIR
    GRAPHS_DIR = f'{DATA_DIR}/GRAPHS/{TAG}_{radii}_{interaction_radius_microns}'
    OUTPUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_Clusters/GNN'
    CONSOL_FILE = f'{DATA_DIR}/{EXP_TAG}.csv'    

    # FEAT = 'ALL_ENVIRON_BAND_CELLTYPE'

        # Model parameters and hyperparameters
    cpu = 'cpu'
    device = 'cuda:0'
    lr = 0.001
    weight_decay = 0.0001
    epochs = 40#3#300#00 # Total number of epochs
    folds =4#4 # number of folds
    scheduler = None
    batch_size = 16
    NEIGH_RADIUS=None#1500
    conv = 'EdgeConv'
    TRAIN_WHOLE = False
    CV = True
    w_model_epochs = 50

    layers = [377]# SHOUD BE THE DIMENTION OF NODE FEATURE VECTOR

    # Loading GRAPHS PATH
    graphlist = glob(os.path.join(GRAPHS_DIR, "*.json"))
    graphDf = pd.DataFrame(graphlist,columns=['Path'])
    graphDf['sample_key'] = [Path(g).stem for g in graphDf['Path']]
    graphDf.set_index('sample_key',inplace=True)

    Exid =  f'FEAT_{FEAT}_lr_{lr}_decay_{weight_decay}_bsize_{batch_size}_layers_{"_".join(map(str,layers))}_dth_{NEIGH_RADIUS}_conv{conv}'
    #Exid = 'TEST_RUN'
             
    end_point = 'sample_type'
    selection = ['Unaffected','MF']
    mapping = {'Unaffected':0, 'MF':1, 'LF':-1, 'INT':-1}
        
    metaDf = pd.read_csv(CONSOL_FILE, index_col='Unnamed: 0')
    mDf = metaDf.loc[:, ['sample', 'patient', end_point]].drop_duplicates().reset_index().set_index('sample')
    consolDf = mDf.join(graphDf)
    trainDf = consolDf[consolDf[end_point].isin(selection)]
    train_graphs = load_graphs(trainDf, mapping=mapping, voi=end_point, features = FEATS_DICT['BAND_DECRIPTOR']+FEATS_DICT['CELL_TYPES'])
    print(f'Total Number of samples {len(train_graphs)}')

    TS = trainDf[[end_point]]

    SS = pd.DataFrame([[float(G.y),G.pid] for G in train_graphs],columns=[end_point,'pid'])
    SS = SS[~SS.pid.duplicated(keep='first')]

    RR = np.full_like(np.zeros((folds, TS.shape[1], 2)),np.nan)
    print('TYPE: ',end_point,
        "# of Positve cases",sum(SS[end_point]==1),' Negative cases ',sum(SS[end_point]==0))
    # Stratified cross validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True)

    Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [
    ], [], [], [], [],  []  # Intialise outputs

    fold = 0
    for trvi, test in skf.split(SS.loc[:,'pid'],SS.loc[:,end_point]):
        
        # selecting Training samples
        train_patients =SS.iloc[trvi,1].tolist()
        valid_patients = SS.iloc[test,1].tolist()
        test_patients = SS.iloc[test,1].tolist()

        # Check for data leakage
        if len(set(train_patients).intersection(set(valid_patients)))>0:
            print('Data Mixing between train test and validation splits')
            exit()

        train_dataset = [train_graphs[i] for i in range(len(train_graphs)) if train_graphs[i].pid in train_patients]
        valid_dataset = [train_graphs[i] for i in range(len(train_graphs)) if train_graphs[i].pid in valid_patients]
        test_dataset = [train_graphs[i] for i in range(len(train_graphs)) if train_graphs[i].pid in test_patients]

        if conv=='PNAConv':
            # Compute the maximum in-degree in the training data.
            deg = compute_degree(train_dataset)
        else:
            deg=0

        v_loader = DataLoader(valid_dataset, shuffle=False)
        tt_loader = DataLoader(test_dataset, shuffle=False)

        model = GNN(dim_features=train_graphs[0].x.shape[1], dim_target=1,
                    degree=deg,
                    layers=layers, dropout=0.1, pooling='mean', conv=conv, aggr='max',
                    device=device)

        net = NetWrapper(model, loss_function=None,
                        device=device,batch_size=batch_size)
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        Q, train_loss, train_acc, val_acc, tt_acc, val_pr, test_pr = net.train(
            train_loader=train_dataset,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=None,#ReduceLROnPlateau(optimizer, 'min'),
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=20,
            return_best=False,
            log_every=5)
        # Fdata.append((best_model, test_dataset, valid_dataset))
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)
        print("\nfold complete", len(Vacc), train_acc,
            val_acc, tt_acc, val_pr, test_pr)

        print('.....'*20,'Saving Convergence Curve','........'*20)

        path_plot_conv = f'{OUTPUT_DIR}/Converg_Curves/{Exid}/{end_point}'
        mkdirs(path_plot_conv)
        import matplotlib.pyplot as plt
        ep_loss = np.array(net.history)
        plt.plot(ep_loss)#[:,0]); plt.plot(ep_loss[:,1]); plt.legend(['train','val']);
        plt.savefig(f'{path_plot_conv}/{len(Vacc)}.png')
        plt.close()

        print('.....'*20,'Saving Best model Weights','........'*20)
        weights_path = f'{OUTPUT_DIR}/Weights/{Exid}/{end_point}'
        mkdirs(weights_path)

        torch.save(Q[0][0].state_dict(), f'{weights_path}/{fold}')

        # Saving node level predictions
        zz, yy, zxn, pn = EnsembleDecisionScoring(
            Q, test_dataset, device=net.device, k=5) # Using 10 ensemble models.

        # saving ensemble results for each fold
        n_classes = zz.shape[-1]
        R = np.full_like(np.zeros((n_classes,2)),np.nan)

        for i in range(n_classes):
            try:
                R[i] = np.array(
                    [calc_roc_auc(yy[:, i], zz[:, i]), calc_pr(yy[:, i], zz[:, i])])
            except:
                print('only one class') 

        df = pd.DataFrame(R, columns=['AUROC', 'AUC-PR'])
        df.index = TS.columns
        RR[fold] = R

        res_dir = f'{OUTPUT_DIR}/Results/{Exid}/{end_point}'
        mkdirs(res_dir)
        df.to_csv(f'{res_dir}/{fold}.csv')
        print(df)

        node_pred_dir = f'{OUTPUT_DIR}/nodePredictions/{Exid}/{end_point}'
        mkdirs(node_pred_dir)

        # saving results of fold prediction
        foldPred = np.hstack((pn[:, np.newaxis], zz, yy))
        foldPredDir = f'{OUTPUT_DIR}/foldPred/{Exid}/{end_point}'
        mkdirs(foldPredDir)

        columns = ['Patient ID'] +[f'P_{voi}' for voi in TS.columns] + [f'T_{voi}' for voi in TS.columns]

        foldDf = pd.DataFrame(foldPred, columns=columns)
        foldDf.set_index('Patient ID', inplace=True)
        foldDf.to_csv(f'{foldPredDir}/{fold}.csv')

        if returnNodeProba:
            for i, GG in enumerate(tqdm(test_dataset)):
                G = Data(edge_index = GG.edge_index,y=GG.y,pid=GG.pid)
                G.to(cpu)
                G.nodeproba = zxn[i][0]
                # adding the target name
                G.class_label = end_point
                ofile = f'{node_pred_dir}/{G.pid}.pkl'
                with open(ofile, 'wb') as f:
                    pickle.dump(G, f)   
        
        # incrementing the fold number
        fold+=1
    # Averaged results of 5 without ensembling
    print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
    print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
    print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
    print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))
    # import pdb; pdb.set_trace()
    import gc; gc.collect()
    RRm = np.nanmean(RR,0)
    RRstd = np.nanstd(RR,0)
    results = pd.DataFrame(np.hstack((RRm, RRstd)))
    results.columns = ['AUROC-mean', 'AUC-PR-mean', 'AUROC-std', 'AUC-PR-std']
    results.index = TS.columns.tolist()
    results.to_csv(f'{OUTPUT_DIR}/Results/{Exid}/{end_point}/{folds}_cv.csv')
    print('Results written to csv on disk')
    print(results)