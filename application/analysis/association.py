import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import mixture
from shutil import copy
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations


import sys
sys.path.append('.')
from application.project import PROJECT_DIR, DATA_DIR, WORKSPACE_DIR, OUTPUT_DIR, CACHE_DIR, EXP_TAG, ANNOT_LEVEL
from application.utils.utilIO import mkdir
ANNOT_LEVEL = 'final_lineage'
# Define constants and paths
K = 3
FIG_SIZE = (8, 10)

PATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull'
OUTPATH = f'{OUTPUT_DIR}/{EXP_TAG}/{ANNOT_LEVEL}/BandFeatures_ConHull_Clusters'


# Main script execution
if __name__ == "__main__":
    TAG = 'EXP_5'
    radii = '1'
    stratification_cols = 'sample_type'
    #stratification_cols = 'disease_status'
    #stratification_cols = 'sample_affect'
    OUT_STATS_FILE = f'{OUTPATH}/br{radii}_{TAG}/dataset_with_envrion.csv'
    OUT_DIR = f'{OUTPATH}/br{radii}_{TAG}/associations_analysis/{stratification_cols}'
    mkdir(OUT_DIR)
    df = pd.read_csv(OUT_STATS_FILE, index_col='Unnamed: 0')
    # Group by sample & ENVIRON and count occurrences
    cell_counts = df.groupby(['sample', 'ENVIRON']).size().reset_index(name='count')

    # Compute relative counts per sample
    cell_counts['relative_count'] = cell_counts.groupby('sample')['count'].transform(lambda x: x / x.sum())

    # Check the stratification (binary or categorical)
    stratification_groups = set(df[stratification_cols])
    all_env = set(df['ENVIRON'].dropna())
    print(f"The stratification column '{stratification_cols}' has {len(stratification_groups)} unique categories: {stratification_groups}")

    # Find exclusive environments for each stratification level
    exclusive_environments = {}
    for group in stratification_groups:
        # Find environments in this category
        env_in_group = set(df[df[stratification_cols] == group]['ENVIRON'].dropna().unique())
        # Find environments that are not in the other categories
        other_groups = stratification_groups - {group}
        env_in_other_groups = set()
        for other_group in other_groups:
            env_in_other_groups.update(df[df[stratification_cols] == other_group]['ENVIRON'].dropna().unique())
        
        # Get exclusive environments for this category
        exclusive_environments[group] = env_in_group - env_in_other_groups
        print(f"Cell types exclusive to {group}: {exclusive_environments[group]}")

    # Pivot without filling NaNs to avoid artificial zeros
    cell_counts_pivot = cell_counts.pivot(index='sample', columns='ENVIRON', values='relative_count')

    # Merge with stratification status
    df_status = df[['sample', stratification_cols]].drop_duplicates()
    df_final = cell_counts_pivot.merge(df_status, on='sample')

    # Melt for visualization
    df_melted = df_final.melt(id_vars=['sample', stratification_cols], var_name='Cell_Type', value_name='Relative_Count')

        # Save the lists of affected samples for each category
    for category, exclusive_envs in exclusive_environments.items():
        df_melted[df_melted['Cell_Type'].isin(exclusive_envs)].to_csv(f"{OUT_DIR}/{category}_exclusive_samples_clusters_props.csv")

    # Boxplot of relative proportions (without artificial zeros)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Cell_Type', y='Relative_Count', hue=stratification_cols)
    plt.xticks(rotation=90)
    plt.title("Relative Cell Type Proportions by Stratification Status")
    plt.ylabel("Relative Count")
    plt.xlabel("Cell Type")
    plt.legend(title="Stratification Status")
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/box_plot_filtered.png')

    # ---- WILCOXON RANK-SUM TEST (Mann-Whitney U) for Binary or Multi-category Stratification ----

    p_values_dict = {}

    # Perform pairwise comparisons for multi-category stratification
    for cell_type in cell_counts_pivot.columns:
        for cat1, cat2 in combinations(stratification_groups, 2):
            # Extract values for each pair
            cat1_vals = cell_counts_pivot[df_final.set_index('sample')[stratification_cols] == cat1][cell_type].dropna()
            cat2_vals = cell_counts_pivot[df_final.set_index('sample')[stratification_cols] == cat2][cell_type].dropna()

            if len(cat1_vals) > 0 and len(cat2_vals) > 0:
                stat, p_val = mannwhitneyu(cat1_vals, cat2_vals, alternative='two-sided')
                p_values_dict[f'{cat1} vs {cat2} - {cell_type}'] = p_val

    # Extract keys and values for FDR correction
    cell_types_tested = list(p_values_dict.keys())
    p_values = list(p_values_dict.values())

    # FDR Correction (Benjamini-Hochberg)
    fdr_results = multipletests(p_values, method='fdr_bh')
    adjusted_p_values = fdr_results[1]

    # Store results in a DataFrame
    stats_df = pd.DataFrame({
        'Cell_Type': cell_types_tested,
        'Raw_p_value': p_values,
        'FDR_Adjusted_p_value': adjusted_p_values
    })

    stats_df.sort_values(by='FDR_Adjusted_p_value', ascending=True)
    # Save statistical results
    stats_df.to_csv(f"{OUT_DIR}/wilcoxon_fdr_results.csv", index=False)

    # Print significant results (FDR < 0.05)
    significant_cells = stats_df[stats_df['FDR_Adjusted_p_value'] < 0.05]
    print("Significant cell types (FDR < 0.05):")
    print(significant_cells)