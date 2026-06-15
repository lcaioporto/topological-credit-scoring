import os
import glob
import pandas as pd
import igraph as ig
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_align_data(parquet_path: str):
    """
    Carrega os dados e replica os filtros do topologicalCS.py 
    para garantir que o índice do DataFrame se alinhe com os nós do Grafo.
    """
    logging.info("Carregando e alinhando dados originais...")
    df = pd.read_parquet(parquet_path)
    
    df = df[df["term60"] == 0].copy()
    
    inconsistent_mask = (df['label'] == 0) & (df['profit'] < 0)
    df = df.loc[~inconsistent_mask].reset_index(drop=True)
    
    df = df.dropna(subset=['num_accts_ever_120_pd', 'mo_sin_rcnt_rev_tl_op'])
    
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    start_date = pd.to_datetime('2012-09-01')
    final_date = pd.to_datetime('2014-03-01')
    df = df[(df['issue_d'] >= start_date) & (df['issue_d'] <= final_date)].reset_index(drop=True)
    
    logging.info(f"Dados alinhados. Total de observações: {len(df)}")
    return df

def evaluate_graph_configurations(graph_dir: str, df_aligned: pd.DataFrame):
    """
    Avalia os grafos focando apenas na janela de treino para evitar Data Leakage.
    """
    # Define a máscara da janela de treino
    train_end_date = pd.to_datetime('2013-09-01')
    train_mask = df_aligned['issue_d'] <= train_end_date
    
    # Extrai os índices (IDs dos nós) e as labels do treino
    train_indices = df_aligned[train_mask].index.tolist()
    train_labels = df_aligned.loc[train_indices, 'label'].astype(int).tolist()
    
    logging.info(f"Janela de treino isolada: {len(train_indices)} nós.")

    search_pattern = os.path.join(graph_dir, "*_lending_club_knn.graphml")
    graph_files = glob.glob(search_pattern)
    
    if not graph_files:
        logging.error("Nenhum arquivo .graphml encontrado.")
        return
    
    results = []
    
    for filepath in graph_files:
        filename = os.path.basename(filepath)
        logging.info(f"Analisando: {filename}")
        
        parts = filename.split('_')
        k_value = parts[0].replace('k', '')
        metric_name = parts[1]
        
        # Carrega o grafo completo
        g_full = ig.Graph.Read_GraphML(filepath)
        
        # Extração do subgrafo de treino (Evita o Data Leakage)
        g_train = g_full.subgraph(train_indices)
        
        # Assortatividade (Homofilia do Target) - O quão bem as labels se agrupam
        # Usamos assortativity_nominal pois a label é categórica (0 ou 1)
        assortativity = g_train.assortativity_nominal(train_labels, directed=True)
        
        # Modularidade do subgrafo de treino (Qualidade da rede)
        g_train_undirected = g_train.copy()
        g_train_undirected.to_undirected()
        communities = g_train_undirected.community_multilevel()
        modularity_score = communities.modularity
        
        # Tamanho da componente gigante no treino
        components = g_train.components(mode='weak')
        giant_ratio = len(components.giant().vs) / g_train.vcount()
        
        results.append({
            'K': int(k_value),
            'Metric': metric_name,
            'Assortativity (Target)': round(assortativity, 4),
            'Modularity (Q)': round(modularity_score, 4),
            'Train_Nodes': g_train.vcount(),
            'Giant_Comp_Ratio': round(giant_ratio, 4)
        })
        
    df_results = pd.DataFrame(results)
    
    # Ordena com peso primário na Assortatividade (maior poder preditivo esperado)
    df_results = df_results.sort_values(
        by=['Assortativity (Target)', 'Modularity (Q)'], 
        ascending=[False, False]
    )
    
    logging.info("\n=== RANKING DAS CONFIGURAÇÕES TOPOLÓGICAS (SOMENTE TREINO) ===")
    print(df_results.to_markdown(index=False))
    return df_results

if __name__ == "__main__":
    PARQUET_PATH = '../data/LCData_accptd-processed.parquet'
    GRAPH_DIR = './graphs_graphml' 
    
    df_aligned = load_and_align_data(PARQUET_PATH)
    evaluate_graph_configurations(GRAPH_DIR, df_aligned)