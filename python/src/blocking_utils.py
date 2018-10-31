import py_entitymatching as em
import csv
import pandas as pd
import configs
import numpy as np

def load_known_dataset(dataset_name):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    l_file_name = dataset_dtls["dataset_folder_path"] + dataset_dtls["ltable_file_name"]
    r_file_name = dataset_dtls["dataset_folder_path"] + dataset_dtls["rtable_file_name"]

    #Assumption: key name is always "id"
    A = em.read_csv_metadata(l_file_name , key="id", encoding='utf-8')
    B = em.read_csv_metadata(r_file_name , key="id", encoding='utf-8')

    return A, B

#Given a dataset name, it returns all duplicates as a Pandas DF
#Assumes the first row is a header
def get_all_duplicates_as_df(dataset_name):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    duplicates_file_name = dataset_dtls["dataset_folder_path"] + dataset_dtls["golden_label_file_name"]
    duplicates_df = pd.read_csv(duplicates_file_name)
    return duplicates_df

#Given a dataset name and candidate set df, this adds a column called gold.
# it is set to  1 for all duplicates and 0 for others
def add_labels_to_candset(dataset_name, candset_df, objectify=False):
    duplicates_df = get_all_duplicates_as_df(dataset_name)
    #rename columns to make merging easier
    duplicates_df.columns = ["ltable_id", "rtable_id"]

    #Sometimes pandas / Magellan puts some columns as objects instead of numeric/string. In this case, we will force this to join appropriately
    if objectify:
        duplicates_df = duplicates_df.astype(object)

    #We merged two DF based on the common attributes. The indicator 'gold' takes three values both, left_only, right_only
    df_with_gold = pd.merge(candset_df, duplicates_df, on=['ltable_id', 'rtable_id'], how='left', indicator='gold')
    #If it is present in both, then it is a duplicate and we set it to 1 and 0 otherwise
    df_with_gold['gold'] = np.where(df_with_gold.gold == 'both', 1, 0)

    #print df_with_gold.loc[df_with_gold['gold']==1,['ltable_id', 'rtable_id']]
    return df_with_gold

def save_candset_ids_only(dataset_name, candset_df):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    output_file_name = dataset_dtls["dataset_folder_path"] + configs.CANDSET_IDS_FILE_NAME
    candset_df[ ['ltable_id', 'rtable_id', 'gold'] ].to_csv(output_file_name, index=False)

def save_candset_compressed(dataset_name, candset_df, file_name):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    output_file_name = dataset_dtls["dataset_folder_path"] + file_name
    candset_df.to_pickle(output_file_name, compression="gzip")

#This function takes the dataset name and block_fn is a function pointer for each dataset used for its blocking
# This stores both the candset in a compressed file and also the ids of candset
def save_candset_wrapper(dataset_name, block_fn, objectify=False):
    A, B = load_known_dataset(dataset_name)
    candset_df = block_fn(A, B)
    candset_with_labels_df = add_labels_to_candset(dataset_name, candset_df, objectify)
    save_candset_compressed(dataset_name, candset_with_labels_df, configs.CANDSET_FILE_NAME)
    save_candset_ids_only(dataset_name, candset_with_labels_df)

def load_candset_compressed(dataset_name, file_name):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    input_file_name = dataset_dtls["dataset_folder_path"] + file_name

    #Magellan complains of missing keys if this is not done!
    ltable, rtable = load_known_dataset(dataset_name)

    #Load compressed file and set Magellan's properties
    candset_df = pd.read_pickle(input_file_name, compression="gzip")

    em.set_key(candset_df,'_id')
    em.set_property(candset_df,'fk_ltable',"ltable_id")
    em.set_property(candset_df,'fk_rtable',"rtable_id")
    em.set_property(candset_df,'ltable',ltable)
    em.set_property(candset_df,'rtable',rtable)

    return candset_df

#assumes the names of foreign keys as ltable_id and rtable_id
def verify_blocking_ground_truth(dataset_name, block_df, objectify=False):
    dataset_dtls = configs.er_dataset_details[dataset_name]
    all_matches_csv_file_name = dataset_dtls["dataset_folder_path"] + dataset_dtls["golden_label_file_name"]
    duplicates_df = pd.read_csv(all_matches_csv_file_name)

    num_duplicates_missed = 0
    duplicates_df.columns = ["ltable_id", "rtable_id"]
    #Sometimes pandas / Magellan puts some columns as objects instead of numeric/string. In this case, we will force this to join appropriately
    if objectify:
        duplicates_df = duplicates_df.astype(object)

    #Intuition: merge function joints two data frames. The outer option creates a number of NaN rows when
    # some duplicates are missing in the blocked_df
    # we leverage the fact that len gives all rows while count gives non-NaN to compute the missing options
    merged_df = block_df.merge(duplicates_df, left_on=["ltable_id", "rtable_id"], right_on=["ltable_id", "rtable_id"], how='outer')
    num_duplicates_missed = len(merged_df) - merged_df["_id"].count()
    total_duplicates = len(duplicates_df)

    print "Size of candset =", len(block_df)
    print "Totally missed:", num_duplicates_missed, " out of ", total_duplicates


def block_fodors_zagats(A, B):
    ob = em.OverlapBlocker()
    #No misses
    C = ob.block_tables(A, B, 'name', 'name', l_output_attrs=['name', 'addr', 'city', 'phone'],  r_output_attrs=['name', 'addr', 'city', 'phone'],
            overlap_size=1, show_progress=False)
    return C


def block_dblp_acm(A, B):
    ab = em.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, l_block_attr='year', r_block_attr='year', l_output_attrs=["title","authors","venue","year"],
		r_output_attrs=["title","authors","venue","year"], allow_missing=True)

    ob = em.OverlapBlocker()

    #=================>results in a candidate set of size 200K with no missing duplicates out of 2224
    #C1 = ob.block_candset(C, 'title', 'title', word_level=True, overlap_size=1, show_progress=True)

    #=================>results in a candidate set of size 46K with 5 missing duplicates out of 2224
    C2 = ob.block_candset(C, 'title', 'title', word_level=True, overlap_size=2, show_progress=True)
    return C2


def block_amazon_googleproducts(A, B):
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 400K with 6 missing duplicates out of 1300
    C = ob.block_tables(A, B, "title", "title", word_level=True, overlap_size=1, l_output_attrs=["title","description","manufacturer","price"], r_output_attrs=["title","description","manufacturer","price"], show_progress=True, allow_missing=True)
    return C


def block_abt_buy(A, B):
    B["description"] = B["description"] + " " + B["manufacturer"]
    ob = em.OverlapBlocker()
    #=================>results in a candidate set of size 164K with 6 missing duplicates out of 1097
    C = ob.block_tables(A, B, "name", "name", word_level=True, overlap_size=1,
	l_output_attrs=["name","description","price"], r_output_attrs=["name","description","price"], show_progress=True, allow_missing=True)
    return C

def block_walmart_amazon(A, B):
    #assumes some preprocessing is done:
    #Specifically in amazon.csv : a.	pcategory2  => groupname , b.	{ proddescrshort,proddescrlong } => shortdescr,longdescr

    ob = em.OverlapBlocker()

    #C1 = ob.block_tables(ltable, rtable, 'title', 'title', word_level=True, overlap_size=2)
    #=================>results in a candidate set of size 1.1M with 20 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C1)

    attributes = ['brand', 'groupname', 'title', 'price', 'shortdescr', 'longdescr', 'imageurl', 'modelno', 'shipweight', 'dimensions']
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 278K with 84 missing duplicates out of 1154
    #blocking_utils.verify_blocking_ground_truth(dataset_name, C2)

    return C2

def block_dblp_scholar(A, B):
    ob = em.OverlapBlocker()
    attributes = ["id","title","authors","venue","year"]
    #C1 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 1.2M with 178 missing duplicates out of 5347
    C2 = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=4, show_progress=True, l_output_attrs=attributes, r_output_attrs=attributes)
    #=================>results in a candidate set of size 135K with 467 missing duplicates out of 5347
    return C2

