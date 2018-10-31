#GiG.

#This file mostly exists as centralized config for various datasets and other things.

DATASET_ROOT = "/Users/neo/Downloads/deeper_civilizer/BenchmarkDatasets/"
#Note: This contains also the label data!
CANDSET_IDS_FILE_NAME = "candset_ids_only.csv"

#TODO: Change me
CANDSET_FILE_NAME = "candset.pkl.compress"

#Size of distributed representations
DR_DIMENSION = 300
FASTTEXT_MODEL_PATH = "/Users/neo/Desktop/QCRI/DataCleaning/datasets/Embeddings/wiki.en/wiki.en.bin"

#Assumptions: partially for interoperability with Magellan/py_entitymatching
# 1. id column is always named as "id"
# 2. all relevant attributes have the same name. any columns without same names will be ignored

er_dataset_details = {
        "Abt_Buy" : {
            "dataset_folder_path" : DATASET_ROOT + "Abt_Buy/",
            "ltable_file_name" : "Abt.csv",
            "rtable_file_name" : "Buy.csv",
            "golden_label_file_name" : "abt_buy_perfectMapping.csv",
        },

        "Amazon_GoogleProducts" : {
            "dataset_folder_path" : DATASET_ROOT + "Amazon_GoogleProducts/",
            "ltable_file_name" : "Amazon.csv",
            "rtable_file_name" : "GoogleProducts.csv",
            "golden_label_file_name" : "Amzon_GoogleProducts_perfectMapping.csv",
        },

        "Walmart_Amazon" : {
            "dataset_folder_path" : DATASET_ROOT + "Walmart_Amazon/",
            "ltable_file_name" : "walmart.csv",
            "rtable_file_name" : "amazon.csv",
            "golden_label_file_name" : "matches_walmart_amazon.csv",
        },

        "DBLP_ACM" : {
            "dataset_folder_path" : DATASET_ROOT + "DBLP_ACM/",
            "ltable_file_name" : "DBLP2.csv",
            "rtable_file_name" : "ACM.csv",
            "golden_label_file_name" : "DBLP-ACM_perfectMapping.csv",
        },

        "DBLP_Scholar" : {
            "dataset_folder_path" : DATASET_ROOT + "DBLP_Scholar/",
            "ltable_file_name" : "DBLP1.csv",
            "rtable_file_name" : "Scholar.csv",
            "golden_label_file_name" : "DBLP-Scholar_perfectMapping.csv",
        },

        "Cora" : {
            "dataset_folder_path" : DATASET_ROOT + "Cora/",
            "ltable_file_name" : "cora.csv",
            "rtable_file_name" : "cora.csv",
            "golden_label_file_name" : "matches_cora.csv",
        },

        "Fodors_Zagat" : {
            "dataset_folder_path" : DATASET_ROOT + "Fodors_Zagat/",
            "ltable_file_name" : "fodors.csv",
            "rtable_file_name" : "zagats.csv",
            "golden_label_file_name" : "matches_fodors_zagats.csv",
        },

}


