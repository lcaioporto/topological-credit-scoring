import pandas as pd
import numpy as np
import igraph as ig
import logging
import time
from collections import Counter
import random

from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from scipy.stats import ks_2samp
import scipy.sparse
from scipy.sparse import issparse, hstack

import sys
sys.path.append('.')
from src.utils.utils import Utils
from generate_graphs import generateGraphs

import shap
import matplotlib.pyplot as plt
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore', category=FutureWarning)

class ModelTrainer:
    def __init__(self, parquet_path, graphml_path):
        self.parquet_path = parquet_path
        self.graphml_path = graphml_path
    
    def load_and_split_data(self):
        logging.info("Carregando e aplicando filtros iniciais do projeto...")

        base_process = generateGraphs(input_path=self.parquet_path, sample=False)
        base_process.presetting_data()
        self.df = base_process.data.copy().reset_index(drop=True)
        
        # Filtro de datas para Treino e Teste
        self.df['issue_d'] = pd.to_datetime(self.df['issue_d'])
        
        train_mask = (self.df['issue_d'] >= '2012-09-01') & (self.df['issue_d'] <= '2013-09-01')
        test_mask = (self.df['issue_d'] >= '2013-10-01') & (self.df['issue_d'] <= '2014-03-01')
        
        self.df_train = self.df[train_mask].copy()
        self.df_test = self.df[test_mask].copy()
        
        self.train_idx = self.df_train.index.tolist()
        self.test_idx = self.df_test.index.tolist()
        
        logging.info(f"Treino (2012-09 a 2013-09): {len(self.df_train)} observações.")
        logging.info(f"Teste (2013-10 a 2014-03): {len(self.df_test)} observações.")

    def extract_topological_features(self):
        logging.info(f"Lendo o grafo: {self.graphml_path}")
        g = ig.Graph.Read_GraphML(self.graphml_path)
        
        # Ancorando IDs originais para não perder o mapeamento ao fazer o subgrafo
        g.vs['orig_id'] = list(range(g.vcount()))
        
        logging.info("========================================")
        logging.info("1. EXTRAÇÃO DE FEATURES: TREINO")
        logging.info("========================================")
        
        # Subgrafo de treino
        g_train = g.subgraph(self.train_idx)
        g_train_undirected = g_train.copy()
        g_train_undirected.to_undirected()
        
        # Features topológicas Não-Supervisionadas do treino
        in_degrees_train = g_train.degree(mode='in')
        pageranks_train = g_train.pagerank()
        clustering_train = g_train_undirected.transitivity_local_undirected(mode="zero")
        
        # Features topológicas Supervisionadas do treino
        train_labels_dict = self.df_train['label'].to_dict()
        r_global_train = self.df_train['label'].mean()
        
        communities = g_train_undirected.community_multilevel()
        membership_train = communities.membership
        
        # Mapeamento do risco real por comunidade no treino
        comm_risk_map = {}
        for c in set(membership_train):
            nodes_in_c = [g_train.vs[v]['orig_id'] for v in range(g_train.vcount()) if membership_train[v] == c]
            labels_in_c = [train_labels_dict[n] for n in nodes_in_c]
            comm_risk_map[c] = np.mean(labels_in_c)
        
        # Mapeamento ID Original -> Comunidade
        orig_to_comm = {g_train.vs[v]['orig_id']: membership_train[v] for v in range(g_train.vcount())}
        
        # Preenchendo as colunas no df_train
        self.df_train['in_degree'] = in_degrees_train
        self.df_train['pagerank'] = pageranks_train
        self.df_train['clustering_coeff'] = clustering_train
        
        neigh_risk_train = []
        comm_risk_train = []
        
        for v in range(g_train.vcount()):
            orig_id = g_train.vs[v]['orig_id']
            # Risco da Comunidade
            comm_risk_train.append(comm_risk_map[orig_to_comm[orig_id]])
            
            # Risco da Vizinhança
            neighbors_v = g_train.successors(v)
            if not neighbors_v:
                neigh_risk_train.append(r_global_train)
            else:
                neighbor_orig_ids = [g_train.vs[n]['orig_id'] for n in neighbors_v]
                neigh_labels = [train_labels_dict[n] for n in neighbor_orig_ids]
                neigh_risk_train.append(np.mean(neigh_labels))
                
        self.df_train['neighborhood_risk'] = neigh_risk_train
        self.df_train['community_risk'] = comm_risk_train


        logging.info("========================================")
        logging.info("2. EXTRAÇÃO DE FEATURES: TESTE")
        logging.info("========================================")
        
        # Como os nós de teste estão no futuro, eles olham para trás (out-edges)
        # e ancoram suas features nos vizinhos que já existiam no treino

        # Imputação das métricas puramente estruturais (evita viés de zeros)
        t_in_degree = np.median(in_degrees_train)
        t_pagerank = np.median(pageranks_train)
        t_clustering = np.median(clustering_train)
        
        neigh_risk_test = []
        comm_risk_test = []
        
        for t_idx in self.test_idx:
            # Pega vizinhos no grafo completo, mas filtra apenas os que estão no treino
            all_neighbors = g.successors(t_idx)
            train_neighbors = [n for n in all_neighbors if n in self.train_idx]
            
            if train_neighbors:
                neigh_labels = [train_labels_dict[n] for n in train_neighbors]
                neigh_risk_test.append(np.mean(neigh_labels))
                
                # Votação de comunidade
                neigh_comms = [orig_to_comm[n] for n in train_neighbors]
                best_comm = Counter(neigh_comms).most_common(1)[0][0]
                comm_risk_test.append(comm_risk_map[best_comm])
            else:
                # O nó de teste se isolou no espaço; aplicamos a média global do passado
                neigh_risk_test.append(r_global_train)
                comm_risk_test.append(r_global_train)
                
        self.df_test['in_degree'] = t_in_degree
        self.df_test['pagerank'] = t_pagerank
        self.df_test['clustering_coeff'] = t_clustering
        self.df_test['neighborhood_risk'] = neigh_risk_test
        self.df_test['community_risk'] = comm_risk_test
        
        logging.info("Extração topológica finalizada com zero data leakage.")
    
    def run_ml_pipeline(self):
        logging.info("Iniciando treinamento dos modelos e cálculo de thresholds...")
        
        dummy_run = generateGraphs(input_path=self.parquet_path, sample=False)
        dummy_run.data = self.df
        numeric_features, ohe_cat_feats, te_cat_feats, binary_features = dummy_run.parsing_data()
        
        preprocessor = Utils.set_preprocessor_pipeline(
            num_feats=numeric_features.tolist(), ohe_cat_feats=ohe_cat_feats,
            te_cat_feats=te_cat_feats, bin_feats=binary_features.tolist()
        )
        
        y_train = self.df_train['label']
        y_test = self.df_test['label']
        profits_train = self.df_train['sim_profit']
        profits_test = self.df_test['sim_profit']
        
        # Processamento
        X_train_tab = preprocessor.fit_transform(self.df_train, y_train)
        X_test_tab = preprocessor.transform(self.df_test)
        
        topo_cols = ['in_degree', 'pagerank', 'clustering_coeff', 'neighborhood_risk', 'community_risk']
        topo_scaler = StandardScaler()
        X_train_topo = topo_scaler.fit_transform(self.df_train[topo_cols])
        X_test_topo = topo_scaler.transform(self.df_test[topo_cols])
        
        if issparse(X_train_tab):
            X_train_hybrid = hstack([X_train_tab, X_train_topo])
            X_test_hybrid = hstack([X_test_tab, X_test_topo])
        else:
            X_train_hybrid = np.hstack((X_train_tab, X_train_topo))
            X_test_hybrid = np.hstack((X_test_tab, X_test_topo))
        
        # Setup dos modelos e tuning
        # Regressão Logística com ElasticNet
        lr_cv_params = {
            "Cs": 10, 
            "cv": 3,  # CV Interno para achar o melhor parâmetro
            "penalty": "l2", 
            "max_iter": 1000, 
            "random_state": 42,
            "n_jobs": -1
        }
        lr_baseline = LogisticRegressionCV(**lr_cv_params)
        lr_hybrid = LogisticRegressionCV(**lr_cv_params)
        
        # XGBoost com RandomizedSearchCV
        xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
        xgb_params_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb_baseline = RandomizedSearchCV(xgb_base, xgb_params_grid, n_iter=10, cv=3, random_state=42, n_jobs=1)
        xgb_hybrid = RandomizedSearchCV(xgb_base, xgb_params_grid, n_iter=10, cv=3, random_state=42, n_jobs=1)

        # Cálculo de thresholds
        logging.info("Calculando predições OOF no treino para LR...")
        preds_train_lr_base = cross_val_predict(lr_baseline, X_train_tab, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
        preds_train_lr_hyb = cross_val_predict(lr_hybrid, X_train_hybrid, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
        
        logging.info("Calculando predições OOF no treino para XGBoost...")
        preds_train_xgb_base = cross_val_predict(xgb_baseline, X_train_tab, y_train, cv=5, method='predict_proba', n_jobs=1)[:, 1]
        preds_train_xgb_hyb = cross_val_predict(xgb_hybrid, X_train_hybrid, y_train, cv=5, method='predict_proba', n_jobs=1)[:, 1]
        
        t_opt_lr_base = self._find_optimal_static_threshold(preds_train_lr_base, profits_train)
        t_opt_lr_hyb = self._find_optimal_static_threshold(preds_train_lr_hyb, profits_train)
        t_opt_xgb_base = self._find_optimal_static_threshold(preds_train_xgb_base, profits_train)
        t_opt_xgb_hyb = self._find_optimal_static_threshold(preds_train_xgb_hyb, profits_train)
        
        # Treinamento final
        logging.info("Treinando modelos finais completos...")
        lr_baseline.fit(X_train_tab, y_train)
        lr_hybrid.fit(X_train_hybrid, y_train)
        xgb_baseline.fit(X_train_tab, y_train)
        xgb_hybrid.fit(X_train_hybrid, y_train)
        
        preds_test_lr_base = lr_baseline.predict_proba(X_test_tab)[:, 1]
        preds_test_lr_hyb = lr_hybrid.predict_proba(X_test_hybrid)[:, 1]
        preds_test_xgb_base = xgb_baseline.predict_proba(X_test_tab)[:, 1]
        preds_test_xgb_hyb = xgb_hybrid.predict_proba(X_test_hybrid)[:, 1]
        
        global_risk = y_train.mean()
        alpha_peso = 1.0 
        
        dyn_t_lr_base = self._calculate_dynamic_thresholds(t_opt_lr_base, self.df_test['community_risk'], global_risk, alpha=alpha_peso)
        dyn_t_lr_hyb = self._calculate_dynamic_thresholds(t_opt_lr_hyb, self.df_test['community_risk'], global_risk, alpha=alpha_peso)
        dyn_t_xgb_base = self._calculate_dynamic_thresholds(t_opt_xgb_base, self.df_test['community_risk'], global_risk, alpha=alpha_peso)
        dyn_t_xgb_hyb = self._calculate_dynamic_thresholds(t_opt_xgb_hyb, self.df_test['community_risk'], global_risk, alpha=alpha_peso)
        
        # Avaliação dos modelos
        logging.info("Iniciando avaliação dos cenários de business...")
        
        # Cenários LR
        self._evaluate_business_scenario("1. LR Normal + Threshold Normal", y_test, preds_test_lr_base, t_opt_lr_base, profits_test)
        self._evaluate_business_scenario("2. LR Normal + Threshold Dinâmico", y_test, preds_test_lr_base, dyn_t_lr_base, profits_test)
        self._evaluate_business_scenario("3. LR Híbrido + Threshold Normal", y_test, preds_test_lr_hyb, t_opt_lr_hyb, profits_test)
        self._evaluate_business_scenario("4. LR Híbrido + Threshold Dinâmico", y_test, preds_test_lr_hyb, dyn_t_lr_hyb, profits_test)
        
        # Cenários XGBoost
        self._evaluate_business_scenario("5. XGB Normal + Threshold Normal", y_test, preds_test_xgb_base, t_opt_xgb_base, profits_test)
        self._evaluate_business_scenario("6. XGB Normal + Threshold Dinâmico", y_test, preds_test_xgb_base, dyn_t_xgb_base, profits_test)
        self._evaluate_business_scenario("7. XGB Híbrido + Threshold Normal", y_test, preds_test_xgb_hyb, t_opt_xgb_hyb, profits_test)
        self._evaluate_business_scenario("8. XGB Híbrido + Threshold Dinâmico", y_test, preds_test_xgb_hyb, dyn_t_xgb_hyb, profits_test)
        
        # SHAP (explicabilidade dos modelos híbridos)
        logging.info("Iniciando geração dos relatórios de explicabilidade...")
        
        try:
            tab_feature_names = preprocessor.get_feature_names_out()
        except Exception:
            tab_feature_names = [f"Feature_{i}" for i in range(X_train_tab.shape[1])]
        
        all_hybrid_features = list(tab_feature_names) + topo_cols
        
        # Explicabilidade Híbrida - Regressão Logística
        self._plot_shap_explanations(
            model=lr_hybrid, X_train=X_train_hybrid, X_test=X_test_hybrid, 
            feature_names=all_hybrid_features, filename="shap_summary_lr_hybrid.png", model_type="linear"
        )
        
        # Explicabilidade Híbrida - XGBoost
        self._plot_shap_explanations(
            model=xgb_hybrid, X_train=X_train_hybrid, X_test=X_test_hybrid, 
            feature_names=all_hybrid_features, filename="shap_summary_xgb_hybrid.png", model_type="tree"
        )

    def _find_optimal_static_threshold(self, y_probs, profits):
        """
        Encontra o limiar (threshold) que maximiza o lucro simulado nos dados de treino.
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t = 0.5
        max_profit = -np.inf
        
        for t in thresholds:
            approved_mask = y_probs < t
            total_profit = profits[approved_mask].sum()
            
            if total_profit > max_profit:
                max_profit = total_profit
                best_t = t
                
        return best_t

    def _calculate_dynamic_thresholds(self, base_t, community_risks, global_risk, alpha=1.0):
        """
        Aplica a fórmula topológica para gerar um threshold personalizado por indivíduo.
        """
        delta_c = community_risks - global_risk
        dynamic_t = base_t - (alpha * delta_c)
        
        # Garante que o threshold se mantenha em um intervalo válido [0.01, 0.99]
        return np.clip(dynamic_t, 0.01, 0.99)

    def _evaluate_business_scenario(self, name, y_true, y_prob, thresholds, profits):
        """
        Avalia as predições utilizando limiares estáticos ou dinâmicos, calculando lucro real.
        """
        # Se thresholds for um array (dinâmico), a comparação é vetorizada elemento a elemento
        approved_mask = y_prob < thresholds
        
        # O modelo previu default (1) se a probabilidade ultrapassou o threshold de aprovação
        y_pred_binary = (~approved_mask).astype(int)
        
        auc = roc_auc_score(y_true, y_prob)
        prob_good = y_prob[y_true == 0]
        prob_bad = y_prob[y_true == 1]
        ks_stat, _ = ks_2samp(prob_good, prob_bad)
        
        total_approved = approved_mask.sum()
        approval_rate = total_approved / len(y_true)
        total_profit = profits[approved_mask].sum()
        
        print(f"\n{'='*55}")
        print(f"Cenário: {name}")
        print(f"{'-'*55}")
        print(f"ROC AUC: {auc:.4f} | KS Stat: {ks_stat:.4f}")
        print(f"Taxa de Aprovação: {approval_rate:.2%}")
        print(f"Lucro da Carteira (Simulado): {total_profit:.2f}")
        print(f"{'-'*55}")
        print(classification_report(y_true, y_pred_binary))

    def _plot_shap_explanations(self, model, X_train, X_test, feature_names, filename="shap_summary.png", model_type="linear"):
        """
        Gera o gráfico de explicabilidade. Suporta 'linear' (Regressão Logística) e 'tree' (XGBoost).
        """
        logging.info(f"Preparando dados para o SHAP ({filename}) [{model_type}]...")
        
        if scipy.sparse.issparse(X_train):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
        else:
            X_train_dense = X_train
            X_test_dense = X_test
            
        if model_type == "linear":
            masker = shap.maskers.Independent(X_train_dense, max_samples=X_train_dense.shape[0])
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_values = explainer.shap_values(X_test_dense)
        elif model_type == "tree":
            # Para o XGBoost encapsulado no RandomizedSearchCV, precisamos extrair o best_estimator_
            best_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_dense)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Gráfico de explicabilidade salvo com sucesso em: {filename}")

if __name__ == "__main__":
    PARQUET_PATH = '../data/LCData_accptd-processed.parquet'
    GRAPH_PATH = 'graphs_graphml/k5_euclidean_lending_club_knn.graphml'
    
    start_time = time.time()
    
    trainer = ModelTrainer(PARQUET_PATH, GRAPH_PATH)
    trainer.load_and_split_data()
    trainer.extract_topological_features()
    trainer.run_ml_pipeline()
    
    elapsed = time.time() - start_time
    logging.info(f"Pipeline concluído em {elapsed/60:.2f} minutos.")