import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing_data import split_raw_data, create_equalized_dataframe
from duplicate_algorithm_logic import create_bin_matrix, start_min_hashing, create_signature_matrix, start_lsh, \
    create_similarity_matrices, compare_kpv

from helper_functions import separate_coordinates_lists, get_band_row_pairs, get_band_row
from evaluation import get_f1_score

NUMBER_OF_BOOTSTRAPS = 100
FRACTION_TO_USE = 0.6
NUMBER_OF_KEYS_TO_USE = 10

current_directory = os.getcwd()
json_path = '\\TVs-all-merged\\TVs-all-merged.json'
path = Path(current_directory + json_path)

with open(path, 'r') as file:
    data_json = json.load(file)

all_products, unique_brand_list, most_occurring_featureMaps_key = split_raw_data(data_json)

equalized_df, selection_most_used_keys = create_equalized_dataframe(all_products, unique_brand_list,
                                                                    most_occurring_featureMaps_key,
                                                                    NUMBER_OF_KEYS_TO_USE)

thresholds = np.arange(0.95, 0.00, -0.05)
thresholds = np.round(thresholds, 2)
row_names = ["pq_lsh", "pc_lsh", "f1_star", "pq", "pc", "f1"]

cumulative_result = pd.DataFrame(data=0, index=row_names, columns=thresholds.astype(str))

for i in range(NUMBER_OF_BOOTSTRAPS):
    print("")
    print(f"Going over bootstrap {i+1}")

    Bootstrap_dataframe = equalized_df.sample(frac=FRACTION_TO_USE).reset_index(drop=True)

    binary_matrix = create_bin_matrix(Bootstrap_dataframe)

    hash_values = start_min_hashing(binary_matrix, round(len(binary_matrix) * 0.5))

    signature_matrix = create_signature_matrix(binary_matrix, hash_values, round(len(binary_matrix) * 0.5))

    band_row_pairs = get_band_row_pairs(signature_matrix)

    result_per_bootstrap = pd.DataFrame(data=0, index=row_names, columns=thresholds.astype(str))

    for threshold_value in thresholds:
        optimal_band_row = get_band_row(band_row_pairs, threshold_value)

        candidate_pairs = start_lsh(signature_matrix, optimal_band_row)

        rows, columns = separate_coordinates_lists(candidate_pairs)

        model_id = Bootstrap_dataframe.iloc[:, 0]
        titles = Bootstrap_dataframe.iloc[:, 1]

        jaccard_lsh, dissim_lsh = create_similarity_matrices(titles, rows, columns)

        result_per_bootstrap = get_f1_score(candidate_pairs, model_id, dissim_lsh, threshold_value,
                                            result_per_bootstrap,
                                            True)

        pairs_msm, dissimilarity_msm = compare_kpv(candidate_pairs, jaccard_lsh, dissim_lsh, equalized_df,
                                                   selection_most_used_keys)

        result_per_bootstrap = get_f1_score(pairs_msm, model_id, dissimilarity_msm, threshold_value,
                                            result_per_bootstrap,
                                            False)

        cumulative_result = + result_per_bootstrap

final_result = cumulative_result / NUMBER_OF_BOOTSTRAPS
final_result.to_csv('ComputerScienceOutput.csv', index=True)

print("Output saved")
