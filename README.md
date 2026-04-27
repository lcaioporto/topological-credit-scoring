# Topological Credit Scoring

This repository contains the source code for a research project that explores the use of graphs to generate topological features from table data with the aim of improving the performance of credit scoring models. The project uses the Lending Club data to execute the experiments.

## Configuring the environment

To configure your environment to run this project, you must initially clone this repository:
```
git clone https://github.com/lcaioporto/topological-credit-scoring.git
```

Then, ensure you have the python version specified in `python-version.txt`, and download the project dependencies:
```
pip install -r requirements.txt
```


## Data preprocessing

### Description
This step downloads the raw data, converts it into a parquet file to compress its size and runs simple transformations on the data to prepare it to run the experiments.

### How to run

First, download the Lending Club dataset. You can do this by running the `Preprocess/download.py` script:
```
python Preprocess/download.py
```
Next, locate the `accepted_2007_to_2018Q4.csv` file in the directory the archives were downloaded and move it to the `../data/` folder (relative to the repository root). To make it easier to handle this data locally, execute the following script to convert the CSV into the Parquet format:
```
python Preprocess/save_as_parquet.py
```
Then, run the preprocessing script to get the final data to be used on the experiments:
```
python Preprocess/prep.py
```

## Running the experiement

After configuring the repository environment, you can execute the experiment to generate the financial graph by running:
```
python src/topologicalCS.py --k 10 --metric euclidean
```
Note that you can specify the value of $K$ and the metric used to calculate distances to build the graph. For instace, we tested with $K: \{5, 10, 30\}$ and $metric: \{euclidean, manhattan, cosine\}$.