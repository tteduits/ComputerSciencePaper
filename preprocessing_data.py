import pandas as pd
import re
from tqdm import tqdm


def split_raw_data(data_json):
    key_occurrences = {}
    all_products = []
    unique_brand_list = ["insignia", "dynex", "tcl", "elite", "viore", "gpx", "contex", "curtisyoung", "mitsubishi",
                         "hiteker", "avue", "optoma"]

    for items in tqdm(data_json.keys(), desc="Splitting data", ):

        for single_item in data_json[items]:
            all_products.append(single_item)
            featuremaps = single_item["featuresMap"]
            featuremaps_brand = featuremaps.get("Brand")

            for key in featuremaps:
                key_occurrences[key] = key_occurrences.get(key, 0) + 1

            if featuremaps_brand is not None:
                featuremaps_brand = featuremaps_brand.lower()
                unique_brand_list.append(featuremaps_brand)
                unique_brand_list = list(set(unique_brand_list))
            else:
                single_item["title"] += " featuresmap_has_no_brand_value"

    del key_occurrences['Brand']
    most_occurring_featuremaps_key = sorted(key_occurrences, key=lambda k: key_occurrences[k], reverse=True)

    return all_products, unique_brand_list, most_occurring_featuremaps_key


def create_equalized_dataframe(all_products, unique_brand_list, most_occurring_featuremaps_key, number_of_keys_used):
    selection_most_used_keys = most_occurring_featuremaps_key[:number_of_keys_used]

    unique_brand_list = sorted(unique_brand_list, key=len, reverse=True)
    equalized_df = pd.DataFrame(columns=[])

    for product in tqdm(all_products, desc="Equalizing title and finding brands"):
        equalized_title, brand = create_equalized_data(product.get("title"), unique_brand_list)

        featuremaps = product["featuresMap"]
        featuremaps_brand = featuremaps.get("Brand")

        if brand is None:
            if featuremaps_brand is not None:
                brand = featuremaps_brand.lower()

        feature_information = {}
        featuremaps = product["featuresMap"]
        for key in selection_most_used_keys:
            feature_item = featuremaps.get(key)
            feature_information.update({key: feature_item})

        equalize_feature_information(feature_information)
        webshop = product["shop"]

        product_information = {**{'ModelID': product.get("modelID"), 'Title': equalized_title, 'Brand': brand,
                                  "FeaturesMaps": featuremaps_brand, "WebShop": webshop}, **feature_information}
        new_row = pd.DataFrame(product_information, index=[0])
        equalized_df = pd.concat([equalized_df, new_row], ignore_index=True)

    # Replace "Pansonic" with "Panasonic" and "lg" with "lg electronics" regardless of case and delete FeatureMaps since it is incomplete
    equalized_df['Brand'] = equalized_df['Brand'].str.replace('pansonic', 'panasonic', case=False)
    equalized_df['Brand'] = equalized_df['Brand'].str.replace('lg', 'lg electronics', case=False)
    equalized_df = equalized_df.drop('FeaturesMaps', axis=1)
    selection_most_used_keys.remove('Component Video Inputs')
    selection_most_used_keys.remove('Aspect Ratio')
    selection_most_used_keys.append('WebShop')
    selection_most_used_keys.append('Brand')

    return equalized_df, selection_most_used_keys


def equalize_feature_information(feature_information):
    equalized_feature_information = {}

    for item in feature_information:
        feature_item = feature_information.get(item)

        if feature_item is None:
            continue

        if isinstance(feature_item, str):
            feature_item = feature_item.lower()
        else:
            feature_item = feature_item.str.lower()

        if item == "Maximum Resolution":
            feature_item = feature_item.replace(" ", "")
            feature_item = feature_item.replace("x", "")
            if feature_item == "1,024768(native)":
                # strange value to "" to avoid error later
                feature_item = ""

        elif item == "Aspect Ratio":
            feature_item = feature_item.replace(":", "")
            feature_item = feature_item.replace(",", "")
            feature_item = feature_item.replace("and", "")
            feature_item = feature_item.replace(" ", "")
            feature_item = feature_item.replace("and", "")
            feature_item = feature_item.replace("|", "")

        elif item == "V-Chip" or item == "USB Port":
            if feature_item.find("yes") != -1:
                feature_item = "yes"
            elif feature_item.find("no") != -1:
                feature_item = "no"
            elif feature_item.find("0") != -1:
                feature_item = "no"
            else:
                feature_item = "no"

        elif item == "Screen Size (Measured Diagonally)" or item == "Screen Size Class" or item == "Vertical Resolution":
            feature_item = re.compile(r'[^0-9]').sub('', feature_item)

        elif item == "TV Type":
            feature_item = re.compile(r'[^a-zA-Z0-9\s]').sub('', feature_item)

        feature_item = feature_item.replace(" ", "")
        feature_information.update({item: feature_item})

    return equalized_feature_information


def create_equalized_data(title_product, unique_brand_list):
    title_product = title_product.lower()
    brand = None

    # Find brand in title only when not found in featuresMap dict
    if title_product.find(" featuresmap_has_no_brand_value") != -1:
        for potential_brand in unique_brand_list:
            if title_product.find(potential_brand) != -1:
                brand = potential_brand

    inches = ["'", '"', "inches", " inches", " inch", "-inch", '”', " '", ' "', ' ”', "-inch"]
    hertzes = ["hertz", " hz", "-hz", " - hz"]

    # Change inch format
    for inch in inches:
        title_product = title_product.replace(inch, "inch")

    # Change hertz format
    for hertz in hertzes:
        title_product = title_product.replace(hertz, "hz")

    regex = re.compile(
        r'(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)[^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))')
    model_word = [x for sublist in regex.findall(title_product) for x in sublist if x != ""]
    model_word = ' '.join(model_word)

    return model_word, brand
