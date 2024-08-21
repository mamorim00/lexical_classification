import nltk
from nltk.corpus import wordnet as wn
import random
import pandas as pd
import tqdm
import numpy as np
from scipy.sparse import lil_matrix  # Add this import
import nltk
from nltk.corpus import wordnet as wn


nltk.download('wordnet')

# Function to build the Hypernym Matrix

def build_hypernym_matrix():
    synsets = list(wn.all_synsets())
    num_synsets = len(synsets)
    hypernym_matrix = np.zeros((num_synsets, num_synsets), dtype=int)

    def add_hypernyms(synset, hypernym_indices):
        for hypernym in synset.hypernyms():
            hypernym_index = synsets.index(hypernym)
            hypernym_indices.add(hypernym_index)
            add_hypernyms(hypernym, hypernym_indices)

    for i, synset in enumerate(synsets):
        print(f"Processing synset {i + 1} of {num_synsets}...")
        hypernym_indices = set()
        add_hypernyms(synset, hypernym_indices)
        for j in hypernym_indices:
            hypernym_matrix[i, j] = 1

    return hypernym_matrix

hypernym_matrix = build_hypernym_matrix()
print(hypernym_matrix)


# Function to build the Holonym Matrix

def build_holonym_matrix():
    synsets = list(wn.all_synsets())
    num_synsets = len(synsets)
    holonym_matrix = lil_matrix((num_synsets, num_synsets), dtype=int)

    # Adiciona as relações diretas de holonimia
    for i in tqdm(range(num_synsets), desc='Progress'):
        synset = synsets[i]
        holonyms = synsets[i].part_holonyms() + synsets[i].substance_holonyms() + synsets[i].member_holonyms()
        for holonym in holonyms:
            j = synsets.index(holonym)
            holonym_matrix[i, j] = 1

    # Adiciona as relações indiretas utilizando a matriz de hiperonimias        
    for i, j in zip(*hypernym_matrix.nonzero()):
        holonyms = synsets[i].part_holonyms() + synsets[i].substance_holonyms() + synsets[i].member_holonyms()
        for h in holonyms:
            k = synsets.index(h)
            holonym_matrix[j, k] = 1

    return holonym_matrix

holonym_matrix = build_holonym_matrix()


# Create the dataframes


# Create an empty DataFrame
df_hypernym = pd.DataFrame(columns=['Definição_Synset', 'ID_Synset', 'Definição_Relacionada', 'ID_Relacionada', 'Relacao'])

# Total number of non-zero elements
total_elements = len(list(zip(*hypernym_matrix.nonzero())))

# Initialize tqdm to track progress
with tqdm(total=total_elements, desc="Processing", unit="element") as pbar:
    # Traverse the non-zero elements of the matrix
    for i, j in zip(*hypernym_matrix.nonzero()):
        # Add the corresponding synsets to the DataFrame
        df_hypernym.loc[len(df_hypernym)] = [synsets[i].definition(), synsets[i].name(), synsets[j].definition(), synsets[j].name(), 'Hypernyms']
        pbar.update(1)  # Update progress bar



# Create an empty DataFrame
df_holonym = pd.DataFrame(columns=['Definição_Synset', 'ID_Synset', 'Definição_Relacionada', 'ID_Relacionada', 'Relacao'])

# Total number of non-zero elements
total_elements = len(list(zip(*holonym_matrix.nonzero())))

# Initialize tqdm to track progress
with tqdm(total=total_elements, desc="Processing", unit="element") as pbar:
    # Traverse the non-zero elements of the matrix
    for i, j in zip(*hypernym_matrix.nonzero()):
        # Add the corresponding synsets to the DataFrame
        df_holonym.loc[len(df_holonym)] = [synsets[i].definition(), synsets[i].name(), synsets[j].definition(), synsets[j].name(), 'Holonyms']
        pbar.update(1)  # Update progress bar

# Unrelated Dataset

# Create an empty DataFrame
df = pd.DataFrame(columns=['Definição_Synset', 'ID_Synset', 'Definição_Relacionada', 'ID_Relacionada', 'Relacao'])

# Total number of non-zero elements
total_elements = 150000

# Initialize tqdm to track progress
with tqdm(total=total_elements, desc="Processing", unit="element") as pbar:
    while(len(df)<150000):
      i = random.randint(0, size-1)
      j = random.randint(0, size-1)
      if holonym_matrix_full[i,j] == 0 and hypernym_matrix_full[i,j]==0:
        df.loc[len(df)] = [synsets[i].definition(), synsets[i].name(), synsets[j].definition(), synsets[j].name(), 'Unrelated']
        pbar.update(1)  # Update progress bar

