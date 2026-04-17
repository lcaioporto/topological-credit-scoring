import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sparse
import logging

# Configure basic logging for tracking the pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialGraphBuilder:
    """
    A class to build and manage K-Nearest Neighbors (KNN) graphs for credit scoring.
    Handles data normalization and memory-efficient sparse matrix generation.
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        """
        Initializes the graph builder.
        
        Args:
            df (pd.DataFrame): The historical loan dataset.
            feature_cols (list): List of column names to be used as node features.
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.features_matrix = self.df[self.feature_cols].values
        self.adjacency_matrix = None
        
        logging.info(f"Initialized GraphBuilder with {self.features_matrix.shape[0]} nodes and {self.features_matrix.shape[1]} features.")

    def _normalize_features(self):
        """
        Imputes missing values using the mean, then applies Standard Scaling 
        (Z-score normalization) to the features. This mirrors the numeric 
        preprocessing pipeline used for the ML models.
        """
        # Impute missing values (NaNs) with the mean of each column
        logging.info("Imputing missing values with column means...")
        imputer = SimpleImputer(strategy='mean')
        self.features_matrix = imputer.fit_transform(self.features_matrix)
        
        # Apply Standard Scaling
        logging.info("Normalizing features using StandardScaler...")
        scaler = StandardScaler()
        self.features_matrix = scaler.fit_transform(self.features_matrix)
        
        logging.info("Imputation and Normalization complete.")
    
    def build_knn_graph(self, k: int = 5, metric: str = 'euclidean', n_jobs: int = -1):
        """
        Constructs the K-NN graph.
        
        Args:
            k (int): Number of nearest neighbors.
            metric (str): Distance metric ('euclidean', 'manhattan', 'cosine', etc.).
            n_jobs (int): Number of parallel jobs (-1 uses all available CPU cores).
            
        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix representing the graph.
        """
        # Always normalize before calculating distances
        self._normalize_features()
        
        logging.info(f"Building K-NN graph using k={k} and metric='{metric}'. This may take a moment...")
        
        # kneighbors_graph automatically uses KD-Tree/Ball-Tree for efficiency
        # mode='connectivity' returns 1s for edges, 0s otherwise. 
        # mode='distance' would return the actual distance weights.
        self.adjacency_matrix = kneighbors_graph(
            X=self.features_matrix, 
            n_neighbors=k, 
            mode='connectivity', 
            metric=metric, 
            n_jobs=n_jobs,
            include_self=False # A node should not be a neighbor to itself in this context
        )
        
        logging.info("Graph construction complete.")
        return self.adjacency_matrix

    def get_graph_metrics(self):
        """
        Calculates basic structural metrics of the generated sparse graph 
        to validate the construction.
        
        Returns:
            dict: Dictionary containing basic graph statistics.
        """
        if self.adjacency_matrix is None:
            raise ValueError("Graph has not been built yet. Call build_knn_graph() first.")
            
        num_nodes = self.adjacency_matrix.shape[0]
        num_edges = self.adjacency_matrix.nnz # Number of non-zero elements (edges)
        
        # In a strict K-NN graph (directed), edges = nodes * K. 
        # If we make it undirected later, this will change.
        sparsity = 1.0 - (num_edges / (num_nodes * num_nodes))
        
        metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "matrix_sparsity": sparsity
        }
        return metrics

    def save_graph(self, filepath: str):
        """
        Saves the sparse adjacency matrix to disk (Community Detection).
        
        Args:
            filepath (str): Path to save the .npz file (e.g., 'data/knn_graph.npz').
        """
        if self.adjacency_matrix is None:
            raise ValueError("No graph to save. Run build_knn_graph() first.")
            
        sparse.save_npz(filepath, self.adjacency_matrix)
        logging.info(f"Sparse graph saved successfully to {filepath}")


if __name__ == "__main__":
    # Mocking the filtered DataFrame (replace with your START_DATE/FINAL_DATE filtered df)
    # Using dummy data to simulate Lending Club features
    mock_data = {
        'loan_amnt': np.random.uniform(1000, 40000, 5000),
        'int_rate': np.random.uniform(5.0, 25.0, 5000),
        'annual_inc': np.random.uniform(30000, 150000, 5000),
        'dti': np.random.uniform(1.0, 40.0, 5000),
        'revol_util': np.random.uniform(0.0, 100.0, 5000)
    }
    df_filtered = pd.DataFrame(mock_data)
    
    # Define the features selected in topologicalCS.py
    selected_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util']
    
    # Instantiate the builder
    graph_builder = FinancialGraphBuilder(df=df_filtered, feature_cols=selected_features)
    
    # Build the graph parametrizing K and the metric
    # Testing with K=10 and Manhattan distance as an example
    adj_matrix = graph_builder.build_knn_graph(k=10, metric='manhattan')
    
    # Extract structural metrics for your partial delivery report
    structural_metrics = graph_builder.get_graph_metrics()
    print("\nGraph Metrics:")
    for key, value in structural_metrics.items():
        print(f"{key}: {value}")
    
    # Save the graph to pass it to the Leiden/Louvain algorithms later
    # graph_builder.save_graph("knn_graph_k10_manhattan.npz")