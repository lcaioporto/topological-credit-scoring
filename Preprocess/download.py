'''
Download the raw Lending Club data from kaggle
'''

import kagglehub

# Download version 3
path = kagglehub.dataset_download("wordsforthewise/lending-club/versions/3")

print("Path to dataset files:", path)