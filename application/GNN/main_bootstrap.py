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
from gutils import load_graphs,pickleLoad
import random

import sys
sys.path.append('.')

import itertools

from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL, FEATS_DICT
from helper_functions import (
    mkdirs,
    update_pidx,
    update_node_predictions,
    generate_statistical_reports,
 )

if __name__ == '__main__':
    
    FEAT = 'BAND_DESCRIPTOR'#'CELL_TYPES_BAND_DECRIPTOR' #'ENVIRON_TYPE' # 'CELL_TYPES'
    #featues_list = FEATS_DICT['CELL_TYPES']# + FEATS_DICT['BAND_DECRIPTOR']#FEATS_DICT['CELL_TYPES'] #+ FEATS_DICT['BAND_DECRIPTOR']
    CELL_COL = 'final_lineage'
    K = 1
    R_max = 300
    TAG = f'K_{K}_R_max_{R_max}'

    interaction_radius_microns = 50
    returnNodeProba = True
    ROOT_DIR = PROJECT_DIR
    GRAPHS_DIR = f'{DATA_DIR}/GRAPHS/{TAG}_{interaction_radius_microns}'

    
    CONSOL_FILE = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_rebuttal/dataset_band_desc_{TAG}.csv'   
    metaDf = pd.read_csv(CONSOL_FILE, index_col='Unnamed: 0')
    OUTPUT_DIR = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_rebuttal/GNN'

    cell_types = metaDf[CELL_COL].unique()
    featues_list = [col for col in metaDf.columns if col.split('_')[0] in cell_types]
    # FEAT = 'ALL_ENVIRON_BAND_CELLTYPE'

    # Model parameters and hyperparameters
    cpu = 'cpu'
    device = 'cuda:0'
    lr = 0.001
    weight_decay = 0.001
    epochs = 40#3#300#00 # Total number of epochs
    n_bootstraps = 20#4#4 # number of n_bootstraps
    scheduler = None
    batch_size = 8
    NEIGH_RADIUS=None#1500
    conv = 'EdgeConv'
    TRAIN_WHOLE = False
    CV = True
    w_model_epochs = 50

    layers = [len(featues_list), len(featues_list)]
              
              #, len(featues_list), len(featues_list), len(featues_list), len(featues_list)]
        #, len(featues_list), len(featues_list), len(featues_list), len(featues_list), len(featues_list), len(featues_list), len(featues_list), len(featues_list),
     #         len(featues_list),len(featues_list), len(featues_list), len(featues_list), len(featues_list)]#,4,4]# SHOUD BE THE DIMENTION OF NODE FEATURE VECTOR
    print(f'GGN Layers', layers)
    print('Experimental Tag ', EXP_TAG)
    # Loading GRAPHS PATH
    graphlist = glob(os.path.join(GRAPHS_DIR, "*.json"))
    graphDf = pd.DataFrame(graphlist,columns=['Path'])
    graphDf['sample_key'] = [Path(g).stem for g in graphDf['Path']]
    graphDf.set_index('sample_key',inplace=True)

    Exid =  f'FEAT_{FEAT}_lr_{lr}_decay_{weight_decay}_bsize_{batch_size}_layers_{"_".join(map(str,layers))}_dth_{NEIGH_RADIUS}_conv{conv}_bootstraps_{n_bootstraps}'
    #Exid = 'TEST_RUN'
             
    end_point = 'sample_type'
    selection = ['Unaffected','MF']
    mapping = {'Unaffected':0, 'MF':1, 'LF':1, 'INT':1}
        
    mDf = metaDf.loc[:, ['sample', 'patient', end_point]].drop_duplicates().reset_index().set_index('sample')
    consolDf = mDf.join(graphDf)
    
    trainDf = consolDf[consolDf[end_point].isin(selection)] #FEATS_DICT['CELL_TYPES']+FEATS_DICT['BAND_DECRIPTOR']
    train_graphs = load_graphs(trainDf, mapping=mapping, voi=end_point, features = featues_list)#FEATS_DICT['BAND_DECRIPTOR']+
    
    #Rest of cases LF and INT
    restDf = consolDf[~consolDf[end_point].isin(selection)]
    rest_graphs = load_graphs(restDf, mapping=mapping, voi=end_point, features = featues_list) 
    
    print(f'Total Number of samples Training {len(train_graphs)}, Rest {len(rest_graphs)}')

    print('Feature set used ', featues_list)

    TS = trainDf[[end_point]]

    #In the cohort a single patient have multiple samples ensuring patient go either to train or test
    SS = pd.DataFrame([[G.sid, G.pid,float(G.y)] for G in train_graphs],columns=['sid','pid', end_point])

    pos_groups = SS[SS[end_point] == 1]['pid'].unique()
    neg_groups = SS[SS[end_point] == 0]['pid'].unique()
    n_pos, n_neg = len(pos_groups), len(neg_groups)

    print('TYPE: ',end_point,
        "# of Positve cases",sum(SS[end_point]==1),' Negative cases ',sum(SS[end_point]==0))

    # Initialize output storage
    all_predictions = [] 
    node_predictions = {}
    auc_pr_results = [] 
    auc_pr_results_combined = [] 
    Vacc, Tacc, Vapr, Tapr = [], [], [], []  # Initialize outputs

    for boot_iter in range(0, n_bootstraps):
        # Stratified Bootstrapping at the patient Level
        train_patients = np.concatenate((
            np.random.choice(pos_groups, size = n_pos, replace=True),
            np.random.choice(neg_groups, size = n_neg, replace=True)
        ))
        #Left out sample used as test set
        test_patients = np.array(list(set(SS.pid) - set(train_patients)))

        # Prepare datasets
        train_dataset = list(itertools.chain.from_iterable(
            [graph] * np.sum(train_patients == graph.pid) for graph in train_graphs if graph.pid in train_patients
        ))
        test_dataset = list(itertools.chain.from_iterable(
            [graph] * np.sum(test_patients == graph.pid) for graph in train_graphs if graph.pid in test_patients
        ))

        # Updating patient ids in the graph as single patients is repeated to be referenced later
        update_pidx(test_dataset)
        update_pidx(train_dataset)
        update_pidx(rest_graphs)

        if conv == 'PNAConv':
            deg = compute_degree(train_dataset)
        else:
            deg = 0

        test_loader = DataLoader(test_dataset, shuffle=False)
        rest_loader = DataLoader(rest_graphs, shuffle=False)

        print(f'Training Model: train dataset counts {len(train_dataset)} test dataset counts {len(test_dataset)}')

        # Initialize Model
        model = GNN(dim_features=train_graphs[0].x.shape[1], dim_target=1,
                    degree=deg, layers=layers, dropout=0.1, pooling='mean',
                    conv=conv, aggr='max', device=device)

        net = NetWrapper(model, loss_function=None, device=device, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train Model
        Q, train_loss, train_acc, val_acc, tt_acc, val_pr, test_pr = net.train(
            train_loader=train_dataset,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=None,
            clipping=None,
            validation_loader=None,  # No validation set in bootstrap
            test_loader=test_loader,
            early_stopping=20,
            return_best=False,
            log_every=5
        )

        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)

        print(f"\nBootstrap iteration {boot_iter+1} complete | Train Acc: {train_acc:.4f}, Test Acc: {tt_acc:.4f}, Test PR: {test_pr:.4f}")
        print(f"Valid AUC: Mean {np.mean(Vacc)}, Median {np.median(Vacc)} Std {np.std(Vacc)}")
        print(f"Test AUC: Mean {np.mean(Tacc)}, Median {np.median(Tacc)} Std {np.std(Tacc)}")
        print(f"Valid PR: Mean {np.mean(Vapr)}, Median {np.median(Vapr)} Std {np.std(Vapr)}")
        print(f"Test PR: Mean {np.mean(Tapr)}, Median {np.median(Tapr)} Std {np.std(Tapr)}")

        print('.....'*20,'Saving Convergence Curve','........'*20)

        path_plot_conv = f'{OUTPUT_DIR}/Converg_Curves/{Exid}/{end_point}'
        mkdirs(path_plot_conv)
        import matplotlib.pyplot as plt
        ep_loss = np.array(net.history)
        plt.plot(ep_loss)#[:,0]); plt.plot(ep_loss[:,1]); plt.legend(['train','val']);
        plt.savefig(f'{path_plot_conv}/{boot_iter}.png')
        plt.close()

        print('.....'*20,'Saving Best model Weights','........'*20)
        weights_path = f'{OUTPUT_DIR}/Weights/{Exid}/{end_point}'
        mkdirs(weights_path)

        torch.save(Q[0][0].state_dict(), f'{weights_path}/{boot_iter}')

        # Getting Test Set Predictions
        zz, yy, zxn, pn = EnsembleDecisionScoring(
            Q, test_dataset, device=net.device, k=5
        ) # Using 5 ensemble models.

        zze, yye, zxne, pne = EnsembleDecisionScoring(
            Q, rest_loader, device=net.device, k=5
        )
        
        # Store results for this bootstrap
        auc_pr_results.append({
            "bootstrap": boot_iter,
            "Test_AUC": calc_roc_auc(yy.flatten(), zz.flatten()),
            "Test_PR": calc_pr(yy.flatten(), zz.flatten())
        })

        yc = np.hstack((yy.flatten(), yye.flatten()))
        zc = np.hstack((zz.flatten(), zze.flatten()))
        pc = np.hstack((pn, pne))

        test_rest_node_pred_dict = dict(zip(pc, zxn+zxne))
        test_rest_graphs = test_dataset + rest_graphs

        auc_pr_results_combined.append({
            "bootstrap": boot_iter,
            "Test_AUC": calc_roc_auc(yc, zc),
            "Test_PR": calc_pr(yc, zc)
        })

        # Save predictions in long format whole cohort
        pred_df = pd.DataFrame({
            "bootstrap": boot_iter,
            "Patient ID": pc,
            "True Label": yc,
            "Predicted Score": zc
        })
        all_predictions.append(pred_df)

        node_pred_dir = f'{OUTPUT_DIR}/nodePredictions/{Exid}/{end_point}'
        mkdirs(node_pred_dir)

        if returnNodeProba:
            node_predictions = update_node_predictions(
                test_rest_graphs, 
                test_rest_node_pred_dict,
                node_predictions, # list is passed repeatedly
                end_point
            )            

    # Convert results to DataFrame
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    auc_pr_results_df = pd.DataFrame(auc_pr_results)

    # Save the final prediction and AUC-PR results
    results_dir = f'{OUTPUT_DIR}/Results/{Exid}/{end_point}'
    mkdirs(results_dir)

    all_predictions_df.to_csv(f'{results_dir}/bootstrap_predictions.csv', index=False)
    auc_pr_results_df.to_csv(f'{results_dir}/auc_pr_results.csv', index=False)
    auc_pr_results_df.to_csv(f'{results_dir}/auc_pr_results_combined.csv', index=False)

    #Generating statistical report
    generate_statistical_reports(all_predictions_df, consolDf, results_dir)

    #Saving the nodeproba graphs as pickle
    pickle_file_path = f'{node_pred_dir}/consol_node_pred_dict.pkl'
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(node_predictions, f)

