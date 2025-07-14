PROJECT_DIR  = '' # Base Project DIR
DATA_DIR = f'{PROJECT_DIR}/data'
WORKSPACE_DIR = f'{PROJECT_DIR}/application'
OUTPUT_DIR = f'{PROJECT_DIR}/OUTPUT'
CACHE_DIR = f'{PROJECT_DIR}/CACHE'

EXP_TAG = 'GSE250346_Seurat_GSE250346_CORRECTED_SEE_RDS_README_082024'
ANNOT_LEVEL = 'final_lineage'
#ANNOT_LEVEL = 'final_CT'

FEATS_DICT = {
    'CELL_TYPES': ['final_lineage_Endothelial', 'final_lineage_Epithelial', 'final_lineage_Immune', 'final_lineage_Mesenchymal'], 
    'ENVIRON_TYPE':['ENVIRON_Endothelial-0', 'ENVIRON_Endothelial-1', 'ENVIRON_Endothelial-2', 'ENVIRON_Endothelial-6', 'ENVIRON_Epithelial-2', 'ENVIRON_Epithelial-3', 'ENVIRON_Immune-0', 'ENVIRON_Immune-2', 'ENVIRON_Immune-5', 'ENVIRON_Immune-6', 'ENVIRON_Immune-8', 'ENVIRON_Immune-9', 'ENVIRON_Mesenchymal-4', 'ENVIRON_Mesenchymal-5', 'ENVIRON_Mesenchymal-8', 'ENVIRON_Mesenchymal-9'],
    'BAND_DECRIPTOR': ['Epithelial_100', 'Epithelial_200', 'Epithelial_300', 'Epithelial_400', 'Epithelial_500', 'Immune_100', 'Immune_200', 'Immune_300', 'Immune_400', 'Immune_500', 'Endothelial_100', 'Endothelial_200', 'Endothelial_300', 'Endothelial_400', 'Endothelial_500', 'Mesenchymal_100', 'Mesenchymal_200', 'Mesenchymal_300', 'Mesenchymal_400', 'Mesenchymal_500']
}

# Python dictionary created with pair-wise zip
keys = [
    "Endothelial-0", "Endothelial-1", "Endothelial-2", "Endothelial-6",
    "Epithelial-2", "Epithelial-3",
    "Immune-0", "Immune-2", "Immune-5", "Immune-6", "Immune-8", "Immune-9",
    "Mesenchymal-4", "Mesenchymal-5", "Mesenchymal-8", "Mesenchymal-9"
]

values = [
    "Endothelial-1", "Endothelial-2", "Endothelial-3", "Endothelial-4",
    "Epithelial-1", "Epithelial-2",
    "Immune-1", "Immune-2", "Immune-3", "Immune-4", "Immune-5", "Immune-6",
    "Mesenchymal-1", "Mesenchymal-2", "Mesenchymal-3", "Mesenchymal-4"
]

# Create dictionary with zip
cluster_rename = dict(zip(keys, values))

selected_clusters = {

    'disease_status': ["Endothelial-1", "Mesenchymal-1", "Immune-1", "Epithelial-2", "Immune-2"],
    'sample_affect': ["Endothelial-1", "Mesenchymal-1", "Immune-1", "Epithelial-2", "Immune-2"]
}