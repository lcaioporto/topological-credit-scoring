'''
From accepted_2007_to_2018Q4.csv, save a parquet file called LCData_accptd.parquet
'''

import polars as pl

schema_overrides = {"id": pl.String}

q = (
    pl.scan_csv(
        '../data/accepted_2007_to_2018Q4.csv', 
        schema_overrides=schema_overrides,
        infer_schema_length=10000,             # Look at more rows to guess types better
        ignore_errors=True
    )
    # Remove the footer rows
    .filter(
        ~pl.col("id").str.contains("Total amount")
    )
)

q.sink_parquet('../data/LCData_accptd.parquet')

print("Conversion complete.")