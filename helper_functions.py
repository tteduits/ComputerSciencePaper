import numpy as np
from nltk import ngrams


def hash_ab(n):
    a_values = []
    b_values = []
    for individual in range(n):
        a_values.append(np.random.randint(1, n))
        b_values.append(np.random.randint(1, n))

    return a_values, b_values


def hash_functions(a, b, x):
    prime = x + 2
    for nr in range(2, prime):
        if prime % nr == 0:
            prime = prime + 1

    hash = (a * x + b) % prime

    return hash


def find_coordinates(signature_matrix):
    coordinates = []

    for i in range(len(signature_matrix)):
        for j in range(len(signature_matrix.columns)):
            if signature_matrix.iloc[i, j] == 1:
                coordinates.append((i, j))

    return coordinates


def separate_coordinates_lists(signature_matrix):
    coordinates = []

    for i in range(len(signature_matrix)):
        for j in range(len(signature_matrix.columns)):
            if signature_matrix.iloc[i, j] == 1:
                coordinates.append((i, j))

    first_coordinates, second_coordinates = zip(*coordinates)

    return list(first_coordinates), list(second_coordinates)


def real_duplicates(vector_model_id):
    number_of_duplicates = 0

    for i in range(len(vector_model_id)):
        for j in range(i + 1, len(vector_model_id) - 1):
            if i == j:
                continue
            if vector_model_id[i] == vector_model_id[j]:
                number_of_duplicates += 1

    return number_of_duplicates


def get_band_row_pairs(signature_matrix):
    b_r_values = []
    n = len(signature_matrix)

    for b in range(1, n + 1):
        r = n // b
        if b * r == n:
            b_r_values.append((b, r))

    return b_r_values


def get_band_row(b_r_values, threshold):
    t_values = []

    for b, r in b_r_values:
        t = (1 / b) ** (1 / r)
        t_values.append(t)

    closest_t = min(t_values, key=lambda x: abs(x - threshold))
    optimal_br_index = t_values.index(closest_t)
    optimal_br = b_r_values[optimal_br_index]

    return optimal_br


def jaccard_similarity(str1, str2, n=2):
    set1 = set(ngrams(str1, n))
    set2 = set(ngrams(str2, n))
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    if union == 0:
        return 1
    else:
        return intersection / union
