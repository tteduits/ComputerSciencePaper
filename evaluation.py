import numpy as np
from helper_functions import real_duplicates


def get_f1_score(candidate_pairs, equalized_df, dissimilarity, threshold_value, result_per_bootstrap, lsh_fase):
    comparisons_made = (candidate_pairs.sum().sum()) / 2
    real_duplicates_number = real_duplicates(equalized_df)
    duplicates_found = 0

    # Filter rows containing inf values
    dissimilarity = dissimilarity.replace([np.inf, -np.inf], np.nan)
    dissimilarity = dissimilarity.dropna(axis=0, how='any')

    for row in range(len(dissimilarity)):
        first_coord = dissimilarity.iloc[row]['first_coord']
        second_coord = dissimilarity.iloc[row]['second_coord']

        if equalized_df.loc[first_coord] == equalized_df.loc[second_coord]:
            duplicates_found += 1

    pq = duplicates_found / comparisons_made
    pc = duplicates_found / real_duplicates_number
    f1_score = (2 * pq * pc) / (pq + pc)

    if lsh_fase:
        result_per_bootstrap.at["pq_lsh", str(threshold_value)] = pq
        result_per_bootstrap.at["pc_lsh", str(threshold_value)] = pc
        result_per_bootstrap.at["f1_star", str(threshold_value)] = f1_score
    else:
        result_per_bootstrap.at["pq", str(threshold_value)] = pq
        result_per_bootstrap.at["pc", str(threshold_value)] = pc
        result_per_bootstrap.at["f1", str(threshold_value)] = f1_score


    return result_per_bootstrap
