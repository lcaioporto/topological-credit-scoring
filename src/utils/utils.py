import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import os

class Utils():
    @staticmethod
    def set_preprocessor_pipeline(
            num_feats: list, ohe_cat_feats: list, te_cat_feats: list, bin_feats: list
            ) -> ColumnTransformer:
        '''
        This pipeline returns a ColumnTransformer containing pipelines for
            - numerical features, with Simple Imputer using the median strategy and Standard Scaler normalization;
            - categorical features, divided on One Hot Encoding and Target Encoding depending on the number of classes;
            - binary features, which is just a passthrough.
        '''
        event_time_feats = [
            'mths_since_last_delinq',         # The number of months since the borrower's last delinquency (49% of NaN)
            'mths_since_last_record',         # The number of months since the last public record          (82% of NaN)
            'mths_since_last_major_derog',    # Months since most recent 90-day or worse rating            (72% of NaN)
            'mths_since_recent_bc_dlq',       # Months since most recent bankcard delinquency              (74% of NaN)
            'mths_since_recent_revol_delinq', # Months since most recent revolving delinquency             (64% of NaN)
            'mths_since_recent_inq'           # Months since most recent inquiry                           (10% of NaN)
        ]
        
        history_time_feats = [
            'emp_length',                     # Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. (6% of NaN)
            'num_tl_120dpd_2m',               # Number of accounts currently 120 days past due (updated in past 2 months)                                                         (4% of NaN)
            'mo_sin_old_il_acct',             # Months since oldest bank installment account opened                                                                               (4% of NaN)
            'percent_bc_gt_75'                # Percentage of all bankcard accounts > 75% of limit                                                                                (1% of NaN)
        ]

        standard_num_feats = [
            f for f in num_feats 
            if f not in event_time_feats and f not in history_time_feats
        ]
        event_time_feats = [f for f in event_time_feats if f in num_feats]
        history_time_feats = [f for f in history_time_feats if f in num_feats]
        
        # Fill with a relatively high value to indicate "Never Happened"
        def custom_max_imputer(arr):
            return np.max(arr)*1.5
        event_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=custom_max_imputer, add_indicator=True)),
            ("scaler", StandardScaler())
        ])

        # Fill with 0 to indicate "No History"
        history_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
            ("scaler", StandardScaler())
        ])

        # Use the median for standard metrics
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ])

        ohe_transformer = Pipeline(
            steps=[
                ("ohe", OneHotEncoder(drop='first', handle_unknown="ignore")),
                ("var_thresh", VarianceThreshold(threshold=1e-4))
            ]
        )

        te_transformer = Pipeline(
            steps = [
                ("te", TargetEncoder(random_state=42))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_event", event_transformer, event_time_feats),
                ("num_hist", history_transformer, history_time_feats),
                ("num_std", numeric_transformer, standard_num_feats),
                ("ohe_cat", ohe_transformer, ohe_cat_feats),
                ("te_cat", te_transformer, te_cat_feats),
                ("bin", "passthrough", bin_feats),
            ]
        )

        return preprocessor

    @staticmethod
    def export_to_graphml(sparse_matrix, save_dir: str, filename: str):
        """
        Exports the sparse matrix to GraphML format.
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"Converting sparse matrix to NetworkX graph...")
        # Convert sparse matrix to a directed NetworkX graph
        G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.DiGraph)
        
        filepath = os.path.join(save_dir, filename)
        print(f"Exporting to {filepath}... (This might take a while for large graphs)")
        nx.write_graphml(G, filepath)
        print("Export complete.")

    @staticmethod
    def analyze_and_plot_topology(sparse_matrix, save_dir="plots", prefix=""):
        """
        Analyzes the topology of the sparse matrix and saves the plots to disk.
        
        Args:
            sparse_matrix: The scipy.sparse matrix representing the graph.
            save_dir (str): The folder path where the images will be saved.
            prefix (str): A string to prepend to the filename (e.g., 'k10_euclidean_').
        """
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        num_nodes = sparse_matrix.shape[0]
        num_edges = sparse_matrix.nnz
        
        # Size and Average Degree
        avg_degree = num_edges / num_nodes
        
        print(f"\n=== Graph Size Metrics ({prefix.strip('_')}) ===")
        print(f"Number of Vertices (Nodes): {num_nodes}")
        print(f"Number of Edges: {num_edges}")
        print(f"Average Degree: {avg_degree:.2f}")
        
        # Degree Distribution (In-Degree)
        in_degrees = np.array(sparse_matrix.sum(axis=0)).flatten()
        
        plt.figure(figsize=(10, 5))
        plt.hist(in_degrees, bins=range(int(max(in_degrees)) + 2), edgecolor='black', alpha=0.7)
        
        # Add prefix to title for better identification if you review them later
        title_suffix = f" ({prefix.strip('_')})" if prefix else ""
        plt.title(f"In-Degree Distribution{title_suffix}")
        
        plt.xlabel("In-Degree (Number of times chosen as a neighbor)")
        plt.ylabel("Frequency (Number of Nodes)")
        plt.grid(axis='y', alpha=0.75)
        
        # Save the plot instead of showing it
        indegree_filename = os.path.join(save_dir, f"{prefix}indegree_distribution.png")
        plt.savefig(indegree_filename, bbox_inches='tight')
        print(f"Saved: {indegree_filename}")
        plt.close()
        
        # Number of Strongly Connected Components
        num_components, component_labels = connected_components(csgraph=sparse_matrix, directed=True, connection='strong')
        print(f"Number of Strongly Connected Components: {num_components}")
        
        # Component Size Distribution
        if num_components > 1:
            unique_labels, counts = np.unique(component_labels, return_counts=True)
            size_frequencies = pd.Series(counts).value_counts().sort_index()
            
            plt.figure(figsize=(10, 5))
            plt.scatter(size_frequencies.index, size_frequencies.values, color='red')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"Component Size Distribution (Log-Log Scale){title_suffix}")
            plt.xlabel("Component Size (Number of Vertices k)")
            plt.ylabel("Frequency (Number of Components with size k)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            
            # Save the plot
            components_filename = os.path.join(save_dir, f"{prefix}component_size_distribution.png")
            plt.savefig(components_filename, bbox_inches='tight')
            print(f"Saved: {components_filename}")
            plt.close()
    
    @staticmethod
    def export_graph_stats_to_md(adj_matrix, save_dir: str, prefix: str):
        """
        Calculates main graph statistics and exports them to a markdown file.
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}graph_stats.md"
        filepath = os.path.join(save_dir, filename)

        # Ensure we are working with a scipy sparse matrix for efficient calculation
        if not sp.issparse(adj_matrix):
            adj_matrix = sp.csr_matrix(adj_matrix)

        # Calculate metrics
        num_vertices = adj_matrix.shape[0]
        num_edges = adj_matrix.nnz
        average_degree = num_edges / num_vertices if num_vertices > 0 else 0.0

        # Calculate strongly connected components
        # connection='strong' calculates SCCs for directed graphs (like k-NN)
        n_components, _ = connected_components(csgraph=adj_matrix, directed=True, connection='strong')

        # Build the markdown content
        md_content = (
            f"- Number of Vertices (Nodes): {num_vertices}\n"
            f"- Number of Edges: {num_edges}\n"
            f"- Average Degree: {average_degree:.2f}\n"
            f"- Number of Strongly Connected Components: {n_components}\n"
        )

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Graph stats successfully saved to {filepath}")