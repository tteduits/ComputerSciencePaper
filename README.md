# ComputerSciencePaper
#The data is imported from a JSON-file.
#The data in the variable title is cleaned (set to lower cases and normalizing frequently used words for inch and hertz).
The model words are extracted and used to make binary vectors each representing a product.
Min-hashing using hashfunctions is performed to create a signature matrix. It reduces sparsity of the binary vector without too much losing information.
All possible combinations of rows and bands, where rows times bands must be the size of the signature matrix, different buckets are formed. With 5 bootstraps of 63% of the data the pair quality, pair completeness and the F1* measure are calculated
The combination of rows and bands that produces the highest value for the F1* measure is run on the complete dataset obtaining candidate duplicates, completing the part where LSH is used as pre-selection method
The dissimilarity matrix is composed. In this matrix the distance between pairs that are not candidate duplicates will be set to inf, the distance between pairs with different completely different values for brand will be set to inf and pairs of product that are from the same webshop will be set to inf. All other candidate pairs, the jaccard sim between the binary vectors are computed.
Our key value classification algorithm is performed where pairs are set to be duplicates if they key value pairs align.
Finaly F1 scores are calculated.
