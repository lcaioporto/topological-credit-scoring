import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

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
    def export_to_graphml(sparse_matrix, filename: str):
        """
        Exports the sparse matrix to GraphML format.
        """
        print(f"Converting sparse matrix to NetworkX graph...")
        # Convert sparse matrix to a directed NetworkX graph
        G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.DiGraph)
        
        print(f"Exporting to {filename}... (This might take a while for large graphs)")
        nx.write_graphml(G, filename)
        print("Export complete.")

    @staticmethod
    def analyze_and_plot_topology(sparse_matrix):
        num_nodes = sparse_matrix.shape[0]
        num_edges = sparse_matrix.nnz
        
        # Size and Average Degree
        # In a directed graph, average degree = |E| / |V|
        avg_degree = num_edges / num_nodes
        
        print("=== Graph Size Metrics ===")
        print(f"Number of Vertices (Nodes): {num_nodes}")
        print(f"Number of Edges: {num_edges}")
        print(f"Average Degree: {avg_degree:.2f}")
        
        # Degree Distribution (In-Degree)
        # Summing columns gives the in-degree (how many times a node was chosen as a nearest neighbor)
        in_degrees = np.array(sparse_matrix.sum(axis=0)).flatten()
        
        plt.figure(figsize=(10, 5))
        plt.hist(in_degrees, bins=range(int(max(in_degrees)) + 2), edgecolor='black', alpha=0.7)
        plt.title("In-Degree Distribution")
        plt.xlabel("In-Degree (Number of times chosen as a neighbor)")
        plt.ylabel("Frequency (Number of Nodes)")
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        
        # Number of Strongly Connected Components
        # connection='strong' is used because the K-NN graph is directed
        num_components, component_labels = connected_components(csgraph=sparse_matrix, directed=True, connection='strong')
        print(f"\nNumber of Strongly Connected Components: {num_components}")
        
        # Component Size Distribution
        if num_components > 1:
            # Count how many nodes are in each component
            unique_labels, counts = np.unique(component_labels, return_counts=True)
            
            # Now count how many components have size 'k'
            size_frequencies = pd.Series(counts).value_counts().sort_index()
            
            plt.figure(figsize=(10, 5))
            # Log scale is often necessary because there is usually one giant component and many tiny ones
            plt.scatter(size_frequencies.index, size_frequencies.values, color='red')
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Component Size Distribution (Log-Log Scale)")
            plt.xlabel("Component Size (Number of Vertices k)")
            plt.ylabel("Frequency (Number of Components with size k)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.show()