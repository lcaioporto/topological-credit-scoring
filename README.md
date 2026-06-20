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

**1. Generating the Financial Graphs**

After configuring the repository environment, you can execute the experiment to generate the financial graph by running:
```
python src/generate_graphs.py --k 10 --metric euclidean
```
Note that you can specify the value of $K$ and the metric used to calculate distances to build the graph. For instace, we tested with $K: \{5, 10, 20\}$ and $metric: \{euclidean, manhattan, cosine\}$.

**2. Evaluating and Selecting the Best Topology**

To prevent data leakage, the generated graphs must be evaluated strictly on the training temporal window. To calculate the Assortativity and Modularity of the generated .graphml files and rank them to find the best configuration, run:
```
python src/select_best_graph.py
```

**3. Training Models and Business Evaluation**

Once the optimal graph topology is identified (e.g., K=5, Euclidean), you can run the machine learning pipeline. This step extracts the topological features, performs nested cross-validation for hyperparameter tuning, calculates the topological dynamic thresholds, and evaluates the final models (Logistic Regression and XGBoost) across multiple business scenarios. To run the pipeline and generate SHAP explanations, execute:
```
python src/run_models.py
```

## Project Structure and File Descriptions

- `generate_graphs.py`: The main orchestrator for graph generation. It handles the initial data filtering, calculates the simulated profit (business target), and calls the graph builder to export .graphml files and summary statistics.

- `build_graph.py`: Contains the FinancialGraphBuilder class. It applies the preprocessing pipelines and constructs K-Nearest Neighbors (K-NN) graphs from the tabular features using scikit-learn, outputting memory-efficient sparse adjacency matrices.

- `select_best_graph.py`: Evaluates the generated graph configurations. It isolates the training nodes to avoid temporal data leakage and ranks the graphs based on Assortativity (target homophily) and Modularity (community quality) using the igraph library.

- `run_models.py`: The comprehensive machine learning pipeline. It extracts complex topological features (such as PageRank and Community Risk) from the selected graph. It trains Logistic Regression and XGBoost models, applies Out-Of-Fold predictions to optimize static thresholds, calculates graph-based dynamic thresholds, outputs financial metrics (simulated profit, ROI, Approval Rates), and generates SHAP (SHapley Additive exPlanations) plots for interpretability.

- `utils.py`: Contains static helper methods. It contains the customized scikit-learn preprocessing pipeline (handling event-based imputations, target encoding, and scaling), GraphML exporters, and plotting functions for topological analysis (in-degree distribution and component sizes).

## Saved Graphs

The `results/` folder contains the generated graphs testing all the 9 combinations of the values mentioned above, as well as their basic statistics, component size distribution and in-degree distribution. We needed to save the larger files using `Git LFS`.