"""
This scripts reads from a data folder, where a parquet file for lending club is assumed to exist.
From a preselected list of columns, it will read the original parquet and save a simplified file.
Some basic transformations are performed, where some binary features are encoded, and numbers represented
as strings are also converted.
A csv file will be created in the root directory with the count of samples for each month.
"""

import polars as pl


def print_value_counts(dat, cols=None):
    """
    Aux function to print value counts for all columns of a polars dataframe.
    It accepts a list of columns of interest as an optional argument.
    """
    if cols is None:
        cols = sorted(dat.columns)
    else:
        cols = [col for col in cols if col in dat.columns]

    for col in cols:
        print(dat.select(pl.col(col).value_counts(sort=True)).unnest(col).collect())
    return None



UTILS_FEATURES = [
    "issue_d",
    "last_pymnt_d",
    "loan_status",
    "total_pymnt",
    "id"
]

NUMERIC_FEATURES = [
    "acc_now_delinq", 
    "acc_open_past_24mths", # NULL UNTIL 2012-03
    # "all_util", # NULL UNTIL 2015-12
    "annual_inc",
    "avg_cur_bal", # NULL UNTIL 2012-08
    "bc_open_to_buy", # NULL UNTIL 2012-03
    "bc_util", # NULL UNTIL 2012-03
    "delinq_2yrs",
    "delinq_amnt",
    "dti",
    "emp_length",
    "fico_range_high",
    "fico_range_low",
    # "inq_fi", # NULL UNTIL 2015-12
    # "inq_last_12m", # NULL UNTIL 2015-12
    "inq_last_6mths",
    "installment",
    "int_rate",
    "loan_amnt",
    # "max_bal_bc", # NULL UNTIL 2015-12
    "mo_sin_old_il_acct", # NULL UNTIL 2012-08
    "mo_sin_old_rev_tl_op", # NULL UNTIL 2012-08
    "mo_sin_rcnt_rev_tl_op", # NULL UNTIL 2012-08
    "mo_sin_rcnt_tl", # NULL UNTIL 2012-08
    "mort_acc", # NULL UNTIL 2012-03
    "mths_since_last_delinq",
    "mths_since_last_major_derog", # NULL UNTIL 2012-08
    "mths_since_last_record",
    # "mths_since_rcnt_il", # NULL UNTIL 2015-12
    "mths_since_recent_bc", # NULL UNTIL 2012-03
    "mths_since_recent_bc_dlq", # NULL UNTIL 2012-08
    "mths_since_recent_inq", # NULL UNTIL 2012-03
    "mths_since_recent_revol_delinq", # NULL UNTIL 2012-03
    "num_accts_ever_120_pd", # NULL UNTIL 2012-08
    "num_actv_bc_tl", # NULL UNTIL 2012-08
    "num_actv_rev_tl", # NULL UNTIL 2012-08
    "num_bc_sats", # NULL UNTIL 2012-06
    "num_bc_tl", # NULL UNTIL 2012-08
    "num_il_tl", # NULL UNTIL 2012-08
    "num_op_rev_tl", # NULL UNTIL 2012-08
    "num_rev_accts", # NULL UNTIL 2012-08
    "num_rev_tl_bal_gt_0", # NULL UNTIL 2012-08
    "num_sats", # NULL UNTIL 2012-06
    "num_tl_120dpd_2m", # NULL UNTIL 2012-08
    "num_tl_30dpd", # NULL UNTIL 2012-08
    "num_tl_90g_dpd_24m", # NULL UNTIL 2012-08
    "num_tl_op_past_12m", # NULL UNTIL 2012-08
    "open_acc",
    # "open_acc_6m", # NULL UNTIL 2015-12
    # "open_il_12m", # NULL UNTIL 2015-12
    # "open_il_24m", # NULL UNTIL 2015-12
    # "open_act_il", # NULL UNTIL 2015-12
    # "open_rv_12m", # NULL UNTIL 2015-12
    # "open_rv_24m", # NULL UNTIL 2015-12
    "pct_tl_nvr_dlq", # NULL UNTIL 2012-08
    "percent_bc_gt_75", # NULL UNTIL 2012-03
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "tax_liens",
    "tot_coll_amt", # NULL UNTIL 2012-08
    "tot_cur_bal", # NULL UNTIL 2012-08
    "tot_hi_cred_lim", # NULL UNTIL 2012-08
    "total_acc",
    "total_bal_ex_mort", # NULL UNTIL 2012-03
    # "total_bal_il", # NULL UNTIL 2015-12
    "total_bc_limit", # NULL UNTIL 2012-03
    # "total_cu_tl", # NULL UNTIL 2015-12
    "total_il_high_credit_limit", # NULL UNTIL 2012-08
    "total_rev_hi_lim", # NULL UNTIL 2012-08
]

CATEGORICAL_FEATURES = [
    "addr_state",
    "sub_grade",
    "purpose",
    "verification_status",
    "home_ownership",
    "grade",
]

BINARY_FEATURES = [
    # "application_type", # UNIQUE VALUE "Induvidual" UNTIL 2015-10
    "initial_list_status",
    "term",
    # "policy_code", # UNIQUE VALUE
]

COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + UTILS_FEATURES + ['title']

STR_SHOULD_BE_NUMS = [
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tot_coll_amt",
    "tot_cur_bal",
    "tot_hi_cred_lim",
]


def prep_data():
    origin_file_path = '../data/LCData_accptd.parquet'

    dat = pl.scan_parquet(origin_file_path)

    modeling_data = (
        dat.filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off"]))
        .select(COLS)
        .with_columns(
            pl.when((pl.col("loan_status") == "Charged Off"))
            .then(1)
            .otherwise(0)
            .alias("label"),

            pl.when(pl.col("term") == " 60 months")
            .then(1)
            .otherwise(0)
            .alias("term60"),

            pl.when(pl.col("initial_list_status") == "f")
            .then(1)
            .otherwise(0)
            .alias("f_initial_list_status"),

            pl.col("revol_util").str.replace("%", "").cast(pl.Float64),

            pl.col("emp_length")
            .str.replace("< 1", "0")
            .str.replace(" years", "")
            .str.replace(" year", "")
            .str.replace("\+", "")
            .cast(pl.Float64),

            (pl.col('total_pymnt') - pl.col('loan_amnt')).alias('profit')
        )
        .with_columns([pl.col(col).cast(pl.Float64) for col in STR_SHOULD_BE_NUMS])
        .drop("loan_status", "term", "initial_list_status", "total_pymnt")
    )
    nrows = modeling_data.select(pl.len()).collect().item()
    low_count_categories = set(
        modeling_data.with_columns(pl.col("title").str.to_lowercase())
        .select(pl.col("title").value_counts(sort=True))
        .collect()
        .unnest("title")
        .filter(pl.col("count") > nrows / 100)["title"]
        .to_list()
    )

    modeling_data = modeling_data.with_columns(
        pl.when(pl.col("title").str.to_lowercase().is_in(low_count_categories))
        .then(pl.col("title"))
        .otherwise(pl.lit("other"))
    )

    # Remove rows which "last_pymnt_d" is null
    modeling_data = modeling_data.filter(pl.col("last_pymnt_d").is_not_null())

    return modeling_data


def main():
    data = prep_data()
    result_file_path = f'../data/LCData_accptd-processed.parquet'
    data.collect().write_parquet(result_file_path)


if __name__ == "__main__":
    main()