import collections
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from helper_functions import hash_ab, hash_functions, jaccard_similarity


def create_bin_matrix(equalized_data):
    model_words = set()

    for title in equalized_data["Title"]:
        model_words.update(title.split())

    model_words_list = sorted(list(model_words))

    binary_matrix = pd.DataFrame(0, index=model_words_list, columns=equalized_data.index)

    for index, product in tqdm(equalized_data.iterrows(), total=len(equalized_data), desc="Creating binary matrix"):
        for mw in model_words_list:
            if mw in product['Title'].split():
                binary_matrix.at[mw, index] = 1

    binary_matrix.columns = binary_matrix.columns + 1

    return binary_matrix


def start_min_hashing(binary_matrix, n):
    rows, columns = np.shape(binary_matrix)

    hash_values = np.zeros((n, rows), dtype=int)

    for row in tqdm(range(rows), desc="Min hashing"):
        a_values, b_values = hash_ab(n)
        for index in range(len(a_values)):
            hash_outcome = hash_functions(a_values[index], b_values[index], n + 1)
            hash_values[index, row] = hash_outcome

    return hash_values


def create_signature_matrix(binary_matrix, hash_values, n):
    rows, columns = binary_matrix.shape
    signature_matrix = pd.DataFrame(np.full((n, columns), np.inf), index=range(n))
    rows_sig, columns_sig = np.shape(signature_matrix)

    for i in tqdm(range(rows), desc="Creating signature matrix"):
        for j in range(columns_sig):
            if binary_matrix.iloc[i, j] != 1:
                continue
            for value in range(len(hash_values)):
                if hash_values[value, i] < signature_matrix.loc[value, j]:
                    signature_matrix.loc[value, j] = hash_values[value, i]

    return signature_matrix


def start_lsh(signature_matrix, band_row_pair):
    n, n_products = signature_matrix.shape
    b = list(band_row_pair)[0]
    bands = np.array_split(signature_matrix, b, axis=0)
    potential_pairs = []

    print("LSH started")
    buckets = collections.defaultdict(set)
    for item, band in enumerate(bands):
        for product in range(n_products):
            band_index = tuple(list(band.iloc[:, product]) + [str(item)])
            buckets[band_index].add(product)

    for potential_pair in buckets.values():
        if len(potential_pair) > 1:
            for pair in itertools.combinations(potential_pair, 2):
                potential_pairs.append(pair)

    potential_pairs = [element for element in potential_pairs if element[0] < element[1]]
    potential_pairs_set = set(potential_pairs)
    candidate_pars = pd.DataFrame(int(0), index=range(n_products), columns=range(n_products))

    for pair in potential_pairs_set:
        candidate_pars.iat[pair[0], pair[1]] = int(1)
        candidate_pars.iat[pair[1], pair[0]] = int(1)

    print("LSH Done")

    return candidate_pars


def create_similarity_matrices(strings, rows, columns):
    jaccard = pd.DataFrame()

    for first_coord, second_coord in zip(rows, columns):
        value = jaccard_similarity(strings[first_coord], strings[second_coord])
        row = {'first_coord': first_coord,
               'second_coord': second_coord,
               'calculated_value': value}
        jaccard = jaccard._append(row, ignore_index=True)

    dissimilarity = jaccard
    dissimilarity['calculated_value'] = 1 - jaccard['calculated_value']

    return jaccard, dissimilarity


def compare_kpv(candidate_pairs, jaccard, dissimilarity, equalized_df, selection_most_used_keys):
    for row in tqdm(range(len(jaccard)), desc="Checking key value pair"):
        first_coord_info = equalized_df.iloc[jaccard['first_coord'].astype(int)]
        second_coord_info = equalized_df.iloc[jaccard['second_coord'].astype(int)]
        for key in selection_most_used_keys:
            if first_coord_info.iloc[row][key] is None or second_coord_info.iloc[row][key] is None:
                continue

            if key == 'WebShop':
                if first_coord_info.iloc[row][key] == second_coord_info.iloc[row][key]:
                    dissimilarity.at[row, 'calculated_value'] = np.inf
                    break

            elif first_coord_info.iloc[row][key] != second_coord_info.iloc[row][key]:
                dissimilarity.at[row, 'calculated_value'] = np.inf
                break

    pairs_msm = candidate_pairs
    for row in range(len(dissimilarity)):
        if dissimilarity.iloc[row, dissimilarity.columns.get_loc('calculated_value')] == np.inf:
            first_coord = int(jaccard.iloc[row, jaccard.columns.get_loc('first_coord')])
            second_coord = int(jaccard.iloc[row, jaccard.columns.get_loc('second_coord')])
            pairs_msm.loc[first_coord + 1, second_coord + 1] = 0
            pairs_msm.loc[second_coord + 1, first_coord + 1] = 0

    return pairs_msm, dissimilarity
