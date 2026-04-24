import pandas as pd
import sys
sys.path.append('.')
from src.utils.utils import Utils
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import time
import numpy as np
from build_graph import FinancialGraphBuilder

# Include flags
import argparse
parser = argparse.ArgumentParser(description="Graph Construction Flags")

parser.add_argument(
    '--sample',
    action='store_true',
    help="If passed, only a small part of the data will be used. Useful for fast testing."
)

TRAIN_TEST_COLUMNS = [
    'acc_now_delinq', 'acc_open_past_24mths',
    'annual_inc', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'delinq_2yrs', 'delinq_amnt', 'dti',
    'emp_length', 'fico_range_high', 'fico_range_low', 'inq_last_6mths',
    'loan_amnt', 'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mort_acc', 'mths_since_last_delinq', 'mths_since_last_major_derog',
    'mths_since_last_record', 'mths_since_recent_bc',
    'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
    'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
    'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
    'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
    'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m', 'open_acc', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'pub_rec', 'pub_rec_bankruptcies', 'revol_bal', 'revol_util',
    'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim',
    'total_acc', 'total_bal_ex_mort', 'total_bc_limit',
    'total_il_high_credit_limit', 'total_rev_hi_lim', 'addr_state',
    'purpose', 'home_ownership', 'title',
    'f_initial_list_status'
    ]

args = parser.parse_args()
SAMPLE = args.sample

if SAMPLE: print("Using only 1% of the original data.")

lr_params = {"C": 1.0, "tol": 1e-4}
MODELS = [("LR", LogisticRegression(**lr_params), 1)]

INPUT_PATH = '../data/LCData_accptd-processed.parquet'
TERM60: bool = 0                          # Selects the contract length of the observations considered in the experiment. Could be either 36 (TERM60 = 0) or 60 (TERM60 = 1)
DEFAULTER_HARM_FACTOR = 1                 # Includes a multiplier to the prejudice caused by defaulters
START_DATE = pd.to_datetime('2012-09-01')
LAST_VALIDATION_DATE = pd.to_datetime('2014-08-01')
FIRST_TEST_DATE = pd.to_datetime('2014-09-01')
FINAL_DATE = pd.to_datetime('2015-03-01')

class topologicalCS():
    def __init__(
            self, input_path: str = INPUT_PATH, models: list = MODELS, sample: bool = SAMPLE, term60: bool = TERM60
        ):
        self.models: list = models
        self.input_path: str = input_path
        self.sample: bool = sample
        self.term60: bool = term60
        self.data: pd.DataFrame = None

    def presetting_data(self):
        '''
        Read the data from the source, drop useless columns, create new features using the existing ones,
        convert columns to pandas date type and select the data by term.
        '''
        # Reading data
        self.data = pd.read_parquet(self.input_path) # 1076751 non-defaulters (label=0), 266246 defaulters (label=1)
        if self.sample: self.data = self.data.sample(frac=0.01, random_state=42)

        # Dropping useless column 'verification_status'
        self.data.drop(columns=['verification_status'], inplace=True)

        # Convert to pandas date type
        self.data['issue_d'] = pd.to_datetime(self.data['issue_d'], format='%b-%Y')
        self.data['last_pymnt_d'] = pd.to_datetime(self.data['last_pymnt_d'], format='%b-%Y')

        # Select by term
        self.data = self.data[self.data["term60"] == self.term60]
        self.data.drop(columns=["term60"], inplace=True)
        
        within_window_data = self.data[(self.data['issue_d'] >= START_DATE) & (self.data['issue_d'] <= LAST_VALIDATION_DATE)]
        good_payers = within_window_data[within_window_data['label'] == 0]
        R = (good_payers['profit'] / (good_payers['loan_amnt'])).mean()
        print(f"The value of R: {R}")
        R = 0.1523 # Mean of profitability of good payers within the period '2012-09' to '2014-05'
        self.data['sim_profit'] = np.where(self.data['label'] == 0, 1.0, -1.0 / R)
        self.data['sim_loan_amnt'] = 1.0

        # Increase the defaulter harm
        if DEFAULTER_HARM_FACTOR != 1:
            self.data['profit'] = np.where(
                self.data['label'] == 1,
                self.data['profit'] * DEFAULTER_HARM_FACTOR,
                self.data['profit']
            )
        self.data['title'] = self.data['title'].str.strip().str.lower()

        # Removing wrongly labeled data (presented negative profit but was considered a good payer)
        inconsistent_mask = (self.data['label'] == 0) & (self.data['profit'] < 0)
        print(f"Dropping {inconsistent_mask.sum()} rows with negative profit marked as good payers.")
        self.data = self.data.loc[~inconsistent_mask].reset_index(drop=True)

        issue_d_censoring_date = self.data['issue_d'].max() # 2018-12-01
        print(f"The last issued loan was registered on: {issue_d_censoring_date.date()}")
        last_pymnt_censoring_date = self.data['last_pymnt_d'].max()
        print(f"The last date captured by the database is (considering max of last_pymnt_d): {last_pymnt_censoring_date.date()}") # 2019-03
        
        # There is a list of 60 observations that present NaN values at all the following features in our experiments period:
        # 'mo_sin_rcnt_tl', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_tl_bal_gt_0',
        # 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_il_high_credit_limit', 'total_rev_hi_lim'
        # To avoid noisy data, these 60 will be removed
        # And there is one unique outlier in which both 'mo_sin_rcnt_rev_tl_op' and 'mo_sin_rcnt_rev_tl_op' are NaN. It will also be removed.
        self.data = self.data.dropna(subset=['num_accts_ever_120_pd', 'mo_sin_rcnt_rev_tl_op'])

        self.data = self.data[(self.data['issue_d'] >= START_DATE) & (self.data['issue_d'] <= FINAL_DATE)]

    def parsing_data(self):
        '''
        Based on predefined criterias, this function parses the data into numeric, binary,
        one hot encoder and target encoder features.
        '''
        # Define binary features
        binary_features = self.data.select_dtypes(include=['int']).columns
        binary_features = binary_features.drop(['label'])

        # Define numeric features
        numeric_features = self.data.select_dtypes(include=['float']).columns
        # Remove 'int_rate', 'profit' and 'installment'
        numeric_features = numeric_features.drop(['int_rate', 'profit', 'installment', 'sim_profit', 'sim_loan_amnt'])

        # Define categorical features
        categorical_features = self.data.select_dtypes(include=['object']).columns
        # Remove 'sub_grade' and 'grade'
        categorical_features = categorical_features.difference(['sub_grade', 'grade'])
        # Less or equal 7 classes - use one hot encoder
        ohe_cat_feats = [col for col in categorical_features if len(self.data[col].unique()) <= 7]
        # More than 7 classes - use target encoder
        te_cat_feats = [col for col in categorical_features if len(self.data[col].unique()) > 7]
        te_cat_feats.remove('id')

        print(f"Numeric features: {numeric_features}")
        print(f"OHE categorical features: {ohe_cat_feats}")
        print(f"TE categorical features: {te_cat_feats}")
        print(f"Binary features: {binary_features}")

        return (numeric_features, ohe_cat_feats, te_cat_feats, binary_features)

    def main(self):
        '''
        This function applies the credit simulation for each model in the list self.models
        '''
        self.presetting_data()
        numeric_features, ohe_cat_feats, te_cat_feats, binary_features = self.parsing_data()
        selected_features = [
            'loan_amnt', 'annual_inc', 'dti', 'revol_util', 'fico_range_low', 'fico_range_high',
            'acc_open_past_24mths', 'inq_last_6mths', 'avg_cur_bal', 'percent_bc_gt_75',
            'mo_sin_old_rev_tl_op'
        ]

        graph_preprocessor = Utils.set_preprocessor_pipeline(
            num_feats=selected_features, 
            ohe_cat_feats=[], 
            te_cat_feats=[], 
            bin_feats=[]
        )

        builder = FinancialGraphBuilder(
            df=self.data,
            feature_cols=selected_features,
            preprocessor=graph_preprocessor
        )

        adj_matrix = builder.build_knn_graph(k=10, metric='euclidean')
        Utils.analyze_and_plot_topology(adj_matrix)
        Utils.export_to_graphml(adj_matrix, "lending_club_knn.graphml")

if __name__ == "__main__":
    # Initialize the time counter
    start_time = time.time()

    run = topologicalCS()
    run.main()
    
    # Finish the time counter
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds, {(elapsed_time/60):.2f} minutes, {(elapsed_time/(60*60)):.2f} hours.")