"""
This module contains functions to process single cell RNA sequencing data.
"""

import json
import os
import re
import shutil
import tarfile
from collections import OrderedDict
from enum import Enum

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import yaml
import zarr
from azure.identity.aio import DefaultAzureCredential
from CancerFinder import CancerFinderInferenceModel
from cellnet.tabnet.tab_network import TabNet
from cellnet.utils.data_loading import dataloader_factory, streamline_count_matrix
from data_integration_utils.azure_blob_manager import OmicBSM
from data_integration_utils.slack_utils import send_message_to_slack
from epigene_tools import BlobStorageManager
from scipy.sparse import csc_matrix
from scipy.stats import median_abs_deviation
from tqdm.auto import tqdm

from constants import (
    BATCH_SIZE_SCTAB,
    CANCERFINDER_0_CLASS,
    CANCERFINDER_1_CLASS,
    CELLTYPE_SCTAB_PATH,
    CHECKPOINT_CANCERFINDER_AZURE_PATH,
    CHECKPOINT_CANCERFINDER_PATH,
    CHECKPOINT_SCTAB_PATH,
    CLASS_CANCERFINDER_COLUMN,
    COUNT_PATTERN,
    DATA_CANCERFINDER_PATH,
    DATA_SCTAB_PATH,
    ENTREZ_ID_COLUMN,
    FOLDER_SCTAB_MLFLOW_PATH,
    GENE_ID_COLUMN,
    GENE_INTER_SIZE,
    GENE_NAME_COLUMN,
    GENES_CANCERFINDER_AZURE_PATH,
    GENES_CANCERFINDER_PATH,
    HPARAMS_SCTAB_PATH,
    MAPPING_SCTAB_PATH,
    MERGED_MODELS_COLUMN,
    MLFLOW_SOURCE_ACCOUNT_URL_VARIABLE,
    MLFLOW_SOURCE_CONTAINER_NAME_VARIABLE,
    OUTPUT_CANCERFINDER_COLUMN,
    PREDICTED_MAPPED_SCTAB_COLUMN,
    UNWANTED_PATTERN,
    VAR_SCTAB_PATH,
)


class ExtSep(Enum):
    TAR_CSV = ","
    TAR_TSV = "\t"
    TAR_TXT = "\t"


def compute_qc_metrics(adata: ad.AnnData, plot: bool = False) -> None:
    """Function that compute multiple QC metrics from scanpy function calculate_qc_metrics and plots several figures
    for future cell filtering.
    We also precise how mitochondrial, ribosomal and hemoglobin are recognized. Be careful, some datasets and gene nomenclature have different
    naming.
    adata reference is updated so we don't need to return the object

    Parameters
    ----------
    adata : AnnData
        global scRNA-Seq data
    plot : bool
        If True, plots figures. Defaults to False.
    adata: ad.AnnData :

    Returns
    -------

    """
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    if plot:
        p1 = sns.displot(adata.obs["total_counts"], bins=100, kde=False)
        p2 = sc.pl.violin(adata, "pct_counts_mt")
        p3 = sc.pl.scatter(
            adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt"
        )


def is_outlier(adata: ad.AnnData, metric: str, nmads: int) -> pd.Series:
    """General function to define outliers in a distribution, based on the number of median absolute deviation (nmad)
    of difference to the median of the distribution.

    Parameters
    ----------
    adata : AnnData
        single cell object
    metric : str
        column of adata to consider for outliers
    nmads : int
        number of median absolute deviation

    Returns
    -------
    pd.Series
        boolean serie that defines which cells are outliers

    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def get_outliers(
    adata: ad.AnnData,
    nmad_total_count: int = 5,
    nmad_n_genes_by_counts: int = 5,
    nmad_pct_counts: int = 5,
    nmad_mito: int = 3,
    thresh_mito: int = 8,
    print_outliers: bool = True,
) -> None:
    """Function that get outliers for multiple specific metrics, log1p_total_counts, log1p_n_genes_by_counts, pct_counts_in_top_20_genes
    and mt_outlier.
    adata reference is updated so the object is not returned.

    Parameters
    ----------
    adata : AnnData
        scRNA-Seq data
    nmad_total_count : int
        number of median absolute deviation for the metric of total count. Defaults to 5.
    nmad_n_genes_by_counts : int
        number of median absolute deviation for the metric of genes by counts. Defaults to 5.
    nmad_pct_counts : int
        number of median absolute deviation for the metric percentage of counts. Defaults to 5.
    nmad_mito : int
        number of median absolute deviation for the metric of percentage of mitochondrial counts. Defaults to 3.
    thresh_mito : int
        hard threshold for percentage of mitochondrial counts. Defaults to 8.
    print_outliers : bool
        If True, prints the number of outliers. Defaults to True.


    Returns
    -------

    """
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", nmad_total_count)
        | is_outlier(adata, "log1p_n_genes_by_counts", nmad_n_genes_by_counts)
        | is_outlier(adata, "pct_counts_in_top_20_genes", nmad_pct_counts)
    )

    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", nmad_mito) | (
        adata.obs["pct_counts_mt"] > thresh_mito
    )
    if print_outliers:
        print(adata.obs["outlier"].value_counts())
        print(adata.obs.mt_outlier.value_counts())


def remove_outliers(adata: ad.AnnData, plot: bool = False) -> ad.AnnData:
    """Remove cells that have been flagged as outliers in at least one metric

    Parameters
    ----------
    adata : AnnData
        scRNA-Seq data
    plot : bool
        If True, plots figures. Defaults to False.

    Returns
    -------
    AnnData
        scRNA-Seq without outliers

    """
    print(f"Total number of cells: {adata.n_obs}")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)]

    print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")
    if plot:
        p1 = sc.pl.scatter(
            adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt"
        )
    return adata


def prepare_data_soupX(adata: ad.AnnData) -> pd.Series:
    """Computes clustering data for soupX background correction

    Parameters
    ----------
    adata : AnnData
        scRNA-Seq data

    Returns
    -------
    pd.Series
        clustering made from uncorrected data

    """
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)

    sc.pp.pca(adata_pp)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="soupx_groups")

    # Preprocess variables for SoupX
    soupx_groups = adata_pp.obs["soupx_groups"]
    del adata_pp
    return soupx_groups


def anndata_to_zarr(
    adata: ad.AnnData, dataset_name: str, omic_silver_manager: OmicBSM
) -> None:
    """Write AnnData to Epigene Zarr format on Azure

    Parameters
    ----------
    adata : AnnData
        AnnData object
    dataset_name : str
        name of the dataset
    omic_silver_manager : OmicBSM
        omic blob storage manager

    Returns
    -------

    """
    credential = DefaultAzureCredential()
    storage_options = {
        "account_name": omic_silver_manager.account_name,
        "account_url": omic_silver_manager.account_url,
        "credential": credential,
    }
    target_filename = omic_silver_manager.get_dataset_filename(
        dataset_name, "scRNA-Seq"
    )

    ad.experimental.write_elem(
        store=zarr.hierarchy.group(
            store=zarr.ABSStore(
                client=omic_silver_manager.container_client,
                prefix=target_filename,
            )
        ),
        elem=adata,
        k="anndata",
    )

    da.to_zarr(
        da.from_array(adata.X)
        .rechunk([adata.shape[0], 25])
        .map_blocks(lambda block: block.todense(), meta=np.array((), dtype=np.float64)),
        url=f"abfs://{omic_silver_manager.container_name}/{target_filename}/dask",
        component="X",
        storage_options=storage_options,
        overwrite=True,
    )


def rename_mitochondrial_genes(
    adata: ad.AnnData, mitochondrial_conversion: dict
) -> None:
    """Rename mitochondrial genes current naming convention

    Parameters
    ----------
    adata : AnnData
        AnnData object
    mitochondrial_conversion : dict
        dictionary of mitochondrial genes

    Returns
    -------

    """
    adata.var.index = adata.var.index.map(
        lambda x: mitochondrial_conversion[x] if x in mitochondrial_conversion else x
    )
    adata.var_names_make_unique()


def store_mitochondrial_distribution(adata: ad.AnnData) -> None:
    """Store mitochondrial distribution in adata.uns

    Parameters
    ----------
    adata : ad.AnnData
        anndata object

    Returns
    -------

    """
    adata.obs["pct_counts_mt"] = adata.obs["pct_counts_mt"].fillna(
        0
    )  # nan because divide by zero if count is null
    mito_df = adata.obs[["sample_id", "pct_counts_mt"]].to_records()
    adata.uns["pct_counts_mt"] = mito_df


# This function is exactly the same as the one used for microarray data
def get_tar_path(list_sup_files: list[str]) -> str:
    """
    Check if supplementary files list contain a tar folder

    Parameters
    ----------
    list_sup_files : list[str]
        list of supplementary files

    Returns
    -------
    str
        TAR folder name

    """
    r = re.compile(r"_RAW\.tar")
    tar_files = list(filter(r.search, list_sup_files))
    if len(tar_files) == 0:
        return None
    else:
        return tar_files[0]


# This function is exactly the same as the one used for microarray data
def extract_molecular_data(sup_file: str, work_dir: str) -> str:
    """Extract files from tar file and remove the .tar archive from folder

    Parameters
    ----------
    sup_file : str
        name of the tar supplementary file
    work_dir : str
        local path where data should be untared

    Returns
    -------
    str
        untared folder name

    """
    tar = tarfile.open(os.path.join(work_dir, os.path.basename(sup_file)), "r:")
    tar.extractall(os.path.join(work_dir, os.path.basename(sup_file).split(".", -1)[0]))
    tar.close()
    os.remove(os.path.join(work_dir, os.path.basename(sup_file)))

    return os.path.basename(sup_file).split(".", -1)[0]


def get_h5_files_prefix(files_list: list[str]) -> list[str]:
    """Get list of h5 files

    Parameters
    ----------
    files_list : list[str]
        list of files

    Returns
    -------
    list[str]
        List of h5 files prefix
    """
    h5_files_prefix = []
    for file in files_list:
        if file.startswith("GSM"):
            h5_files_prefix.append(file.split(".")[0])
    return list(set(h5_files_prefix))


# check the kind of files in the tar
def check_file_type(file_list: list[str]):
    """
    Check if couple of genes/barcode, feature.barcodes files are in the folder and return file prefix accordingly

    Parameters
    ----------
    file_list : list[str]
        List of files

    Returns
    -------
    list[str], list[str]
        List of files containing genes pattern, List of files containing features pattern

    """
    genes_sample_pattern = [
        fl.split("genes")[0]
        for fl in file_list
        if ("genes.tsv" in fl) & (fl.startswith("GSM"))
    ]
    features_sample_pattern = [
        fl.split("features")[0]
        for fl in file_list
        if ("features.tsv" in fl) & (fl.startswith("GSM"))
    ]
    return genes_sample_pattern, features_sample_pattern


def check_no_intersection_pattern_list(
    genes_sample_pattern: list[str], features_sample_pattern: list[str]
) -> bool:
    """
    Check if there is no intersection between gene_sample_pattern and features_sample_pattern

    Parameters
    ----------
    genes_sample_pattern : list[str]
        List of genes sample pattern
    features_sample_pattern : list[str]
        List of features sample pattern

    Returns
    -------
    boolean: True if there is no intersection, False otherwise
    """
    return (
        len(set(genes_sample_pattern).intersection(set(features_sample_pattern))) == 0
    )


def unzip_genes_barcode_matrix_files_triplets(
    work_dir: str, files_list: list[str], pattern_list: list[str]
) -> None:
    """
    Unzip genes/barcode/matrix files triplet for samples detected by the file type check. File are unziped in the work_dir

    Parameters
    ----------
    work_dir : str
        Path to the working directory
    files_list : list[str]
        List of files
    pattern_list : list[str]
        List of patterns
    """
    files_to_process = [
        file
        for file in files_list
        if any([pattern in file for pattern in pattern_list])
    ]
    for file in files_to_process:
        if file.endswith(".gz"):
            os.system(f"gunzip {work_dir}/{file}")


def list_files_RAW(data_path: str, dataset: str) -> list[str]:
    """
    List files in the GSEXXX_RAW folder

    Parameters
    ----------
    data_path: str
        Path to the data folder
    dataset: str
        Dataset name

    Returns
    -------
    list:
        list of files in the GSEXXX_RAW folder
    """
    return os.listdir(os.path.join(data_path, f"{dataset}_RAW"))


def check_3_files_per_sample_prefix(
    prefix_list: list[str], files_list: list[str]
) -> bool:
    """
    Check if there are 3 files per sample prefix

    Parameters
    ----------
        prefix_list : list[str]
            list of prefixes
        files_list : list[str]
            list of files

    Returns
    -------
        boolean: True if there are 3 files per prefix, False otherwise
    """
    nb_file_per_prefix = np.unique(
        [" ".join(files_list).count(pattern) for pattern in prefix_list]
    )
    return len(nb_file_per_prefix) == 1 & (nb_file_per_prefix == 3)


def combine_single_cell_data(
    file_prefixes: list[str],
    data_path: str,
    dataset: str,
    sc_file_type: str,
    splitted_reference: dict,
) -> ad.AnnData:
    """Combine single cell data from multiple samples into a single AnnData object

    Parameters
    ----------
    file_prefixes : list[str])
        list of sample prefixes
    data_path : str
        path to the data folder
    dataset : str
        dataset name
    sc_file_type : str
        type of file in which information are stored

    Returns
    -------
    ad.AnnData
        anndata object containing all the samples
    """
    adata_list = []
    for prefixe in file_prefixes:
        if sc_file_type == "TAR_MTX":
            adata = sc.read_10x_mtx(
                os.path.join(data_path, f"{dataset}_RAW"), prefix=f"{prefixe}"
            )
        elif sc_file_type == "TAR_H5":
            adata = sc.read_10x_h5(
                os.path.join(data_path, f"{dataset}_RAW", f"{prefixe}.h5")
            )
        elif (
            sc_file_type == "TAR_CSV"
            or sc_file_type == "TAR_TSV"
            or sc_file_type == "TAR_TXT"
        ):
            molecular_data = pd.read_csv(
                os.path.join(data_path, f"{dataset}_RAW", f"{prefixe}"),
                index_col=0,
                sep=ExtSep[sc_file_type].value,
            )
            # check the samples are in the columns and transpose the data if necessary
            molecular_data = detect_samples_columns(molecular_data)
            molecular_data = molecular_data.reset_index()
            # check the gene names/id are detected and correspond to the data index
            molecular_data, gene_column_found = find_rename_reindex_gene_column(
                molecular_data, splitted_reference
            )
            if not gene_column_found:
                raise ValueError("Gene column not found. Cannot pre-process the data")

            adata = ad.AnnData(molecular_data.T)

        else:
            raise ValueError("File format not supported")
        adata.var_names_make_unique()
        adata.obs["dataset"] = dataset
        adata.obs["sample_id"] = prefixe.split("_")[0]
        adata.obs.index = adata.obs["sample_id"] + "_" + adata.obs.index
        adata.obs.index.name = None
        adata.var.index.name = None
        adata_list.append(adata)
    adata_combined = ad.concat(adata_list, join="inner", merge="same")
    return adata_combined


def filter_raw_count_files(file_list) -> list[str]:
    """Filter files to keep only raw count based on file names

    Args:
        file_list (list): list of files in the dataset
    Returns:
        list: list of raw count files
    """
    raw_file = [file for file in file_list if "raw" in file]
    if len(raw_file) == 0:
        raw_file = [
            file
            for file in file_list
            if any(pattern in file.lower() for pattern in COUNT_PATTERN)
        ]

    raw_file = [
        file
        for file in raw_file
        if all(pattern not in file.lower() for pattern in UNWANTED_PATTERN)
    ]
    return raw_file


def detect_samples_columns(sc_dataframe: pd.DataFrame):
    """Detect if samples are in the column or index of the dataframe. Cell barcode for 10X data is 16bp long and contains only ATGC.

    Args:
        sc_dataframe (pd.DataFrame): single cell data
    """

    matching_val_col, matching_count_col = np.unique(
        sc_dataframe.columns.str.match("^.*[ATGC]{16}.*"),
        return_counts=True,
    )
    matching_val_index, matching_count_index = np.unique(
        sc_dataframe.index.str.match("^.*[ATGC]{16}.*"),
        return_counts=True,
    )
    if (
        len(matching_val_col) == 1
        and (matching_val_col[0])
        and (matching_count_col[0] == sc_dataframe.shape[1])
    ):
        # samples are in the columns, return the datafarme as is
        return sc_dataframe
    elif (
        len(matching_val_index) == 1
        and (matching_val_index[0])
        and (matching_count_index[0] == sc_dataframe.shape[0])
    ):
        # samples are in the index, return the transposed datafarme
        return sc_dataframe.T
    else:
        raise ValueError(
            "Samples are not detected in the column or index of the dataframe"
        )


def split_gene_standardisation_by_reference(
    gene_standardisation_df: pd.DataFrame,
) -> dict:
    """Split gene standardisation dataframe by reference.

    Parameters
    ----------
    gene_standardisation_df : pd.DataFrame
        gene standardisation dataframe

    Returns
    -------
    dict
        dictionary of gene standardisation dataframes by reference

    """
    gene_standardisation_by_reference = {}
    gene_standardisation_by_reference[GENE_NAME_COLUMN] = (
        gene_standardisation_df[
            (gene_standardisation_df["relation"] == "Gene_name")
            & (gene_standardisation_df["source"].str.contains("gencode"))
        ]["gene_name"]
        .str.lower()
        .values.tolist()
    )
    gene_standardisation_by_reference[ENTREZ_ID_COLUMN] = (
        gene_standardisation_df[
            gene_standardisation_df["gene_name"].apply(
                lambda x: x.replace(".", "").isnumeric()
            )
        ]["gene_name"]
        .astype(float)
        .astype(int)
        .astype(str)
        .values.tolist()
    )
    gene_standardisation_by_reference[GENE_ID_COLUMN] = (
        gene_standardisation_df[
            gene_standardisation_df["gene_name"].str.startswith("ENSG")
        ]["gene_name"]
        .str.split(".")
        .str[0]
        .str.lower()
        .unique()
        .tolist()
    )
    return gene_standardisation_by_reference


def find_rename_reindex_gene_column(
    data: pd.DataFrame, splitted_reference: dict
) -> tuple[pd.DataFrame, bool]:
    """Function to find gene patterns and define column to consider

    Parameters
    ----------
    data : pd.DataFrame
        expression data for one sample
    splitted_reference : dict
        dictionary with gene reference information

    Returns
    -------
    tuple[pd.DataFrame, bool]
        cleaned data with renamed column and
        boolean for finding gene column

    """
    data_subset = data.select_dtypes(exclude="number")
    data_size = data.shape[0]
    for column in data_subset:
        cleaned_column = set.union(
            *[
                set(ll)
                for ll in data[column]
                .astype(str)
                .str.lower()
                .str.split(r"[.|_]")
                .values
            ]
        )
        if (
            (
                gene_id_inter := len(
                    cleaned_column.intersection(set(splitted_reference[GENE_ID_COLUMN]))
                )
            )
            > data_size / 2
        ) | (gene_id_inter > GENE_INTER_SIZE):
            if GENE_ID_COLUMN in data.columns and column != GENE_ID_COLUMN:
                data = data.rename(columns={GENE_ID_COLUMN: f"{GENE_ID_COLUMN}_1"})
            return (
                data.dropna(subset=[column])
                .rename(columns={column: GENE_ID_COLUMN})
                .set_index(GENE_ID_COLUMN),
                True,
            )
        elif (
            (
                gene_name_inter := len(
                    cleaned_column.intersection(
                        set(splitted_reference[GENE_NAME_COLUMN])
                    )
                )
            )
            > data_size / 2
        ) | (gene_name_inter > GENE_INTER_SIZE):
            if GENE_NAME_COLUMN in data.columns and column != GENE_NAME_COLUMN:
                data = data.rename(columns={GENE_NAME_COLUMN: f"{GENE_NAME_COLUMN}_1"})
            return (
                data.dropna(subset=[column])
                .rename(columns={column: GENE_NAME_COLUMN})
                .set_index(GENE_NAME_COLUMN),
                True,
            )
        elif (
            (
                gene_entrez_inter := len(
                    cleaned_column.intersection(
                        set(splitted_reference[ENTREZ_ID_COLUMN])
                    )
                )
            )
            > data_size / 2
        ) | (gene_entrez_inter > GENE_INTER_SIZE):
            if ENTREZ_ID_COLUMN in data.columns and column != ENTREZ_ID_COLUMN:
                data = data.rename(columns={ENTREZ_ID_COLUMN: f"{ENTREZ_ID_COLUMN}_1"})
            return (
                data.dropna(subset=[column])
                .rename(columns={column: ENTREZ_ID_COLUMN})
                .set_index(ENTREZ_ID_COLUMN),
                True,
            )
        else:
            continue

    return data, False


def sf_log1p_norm(celldata_raw):
    """Normalize single cell data by scaling to 10,000 counts per cell and applying log1p transform.

    This function performs two key preprocessing steps commonly used in single cell RNA-seq analysis:
    1. Normalizes each cell to have 10,000 total counts
    2. Applies log(x+1) transformation to stabilize variance

    Parameters
    ----------
    x : torch.Tensor
        Input tensor containing gene expression data where rows are cells and columns are genes

    Returns
    -------
    torch.Tensor
        Normalized and log1p transformed expression data
    """

    counts = torch.sum(celldata_raw, dim=1, keepdim=True)
    # avoid zero division error
    counts += counts == 0.0
    scaling_factor = 10000.0 / counts

    return torch.log1p(scaling_factor * celldata_raw)


def label_mapping(predictions: pd.DataFrame, output_mapping: dict[str, str]):
    """Standardize and map cell type labels to a common ontology.

    This function performs maps on predicted cell types to a standardized ontology using the provided mapping.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing cell type predictions with at least a 'predicted' column
    output_mapping : dict[str, str]
        Dictionary mapping original cell type names to standardized names

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for mapped and standardized cell types
    """
    # Map the predictions to the ontology
    predictions[PREDICTED_MAPPED_SCTAB_COLUMN] = predictions["predicted"].map(
        output_mapping
    )

    return predictions


def execute_cancer_finder(adata_combined: ad.AnnData, slack_conf):
    """Execute CancerFinder model to predict cancer status of cells in the dataset.

    This function:
    1. Downloads necessary model files if not present locally
    2. Loads the CancerFinder model and gene list
    3. Preprocesses the input data
    4. Makes predictions on the data
    5. Adds prediction results to the AnnData object's obs dataframe

    Parameters
    ----------
    adata_combined : AnnData
        Combined AnnData object containing all samples
    slack_conf : dict
        Configuration for Slack notifications

    Returns
    -------
    None
        Results are added directly to adata_combined.obs
    """
    if not os.path.exists(DATA_CANCERFINDER_PATH):
        os.mkdir(f"{DATA_CANCERFINDER_PATH}/")

        blob_manager = BlobStorageManager(
            container_name=os.environ[MLFLOW_SOURCE_CONTAINER_NAME_VARIABLE],
            account_url=os.environ[MLFLOW_SOURCE_ACCOUNT_URL_VARIABLE],
        )

        blob_manager.download_file(
            CHECKPOINT_CANCERFINDER_AZURE_PATH,
            destination_path=CHECKPOINT_CANCERFINDER_PATH,
        )
        blob_manager.download_file(
            GENES_CANCERFINDER_AZURE_PATH,
            destination_path=GENES_CANCERFINDER_PATH,
        )

    with open(GENES_CANCERFINDER_PATH) as f:
        gene_list = json.load(f)

    qtd_genes = len(gene_list)
    set_intersection = set(list(adata_combined.var.index)).intersection(set(gene_list))
    intersection = len(set_intersection)
    intersection_pct = intersection/qtd_genes
    list_intersection = list(set_intersection)

    send_message_to_slack(
        channel=slack_conf["channel"],
        slack_webhook_url=slack_conf["webhook_url"],
        message=f":dna: [*Single cell pipeline*] \n The CancerFinder has {qtd_genes} genes and the dataset has {intersection} of them. \n",
    )
    # Loading the model to make the predictions
    cf_model = CancerFinderInferenceModel()
    cf_model.fit(matrix=None, threshold=0.5, ckp=CHECKPOINT_CANCERFINDER_PATH)

    cf_model.load_context()

    matrix_preprocessed = cf_model.preprocess(adata_combined, gene_list)

    output = cf_model.predict_logic(context=None, model_input=matrix_preprocessed)

    adata_combined.obs = adata_combined.obs.merge(
        output, left_index=True, right_on="sample", how="left"
    )

    adata_combined.obs = adata_combined.obs.rename(
        columns={"predict": OUTPUT_CANCERFINDER_COLUMN}
    )

    adata_combined.obs[CLASS_CANCERFINDER_COLUMN] = (
        adata_combined.obs[OUTPUT_CANCERFINDER_COLUMN]
        .map({1: CANCERFINDER_1_CLASS})
        .fillna(CANCERFINDER_0_CLASS)
    )
    return adata_combined, intersection_pct, list_intersection


def execute_sctab(adata_combined: ad.AnnData):
    """Execute scTab model to predict cell types in the dataset.

    This function:
    1. Downloads necessary model files if not present locally
    2. Loads the scTab model, gene list, and cell type mappings
    3. Preprocesses and normalizes the input data
    4. Makes predictions using the TabNet model
    5. Maps predictions to standardized cell type labels
    6. Adds prediction results to the AnnData object's obs dataframe

    Parameters
    ----------
    adata_combined : AnnData
        Combined AnnData object containing all samples

    Returns
    -------
    None
        Results are added directly to adata_combined.obs
    """
    if not os.path.exists(DATA_SCTAB_PATH):
        os.makedirs(
            f"{FOLDER_SCTAB_MLFLOW_PATH}/scTab-checkpoints",
            exist_ok=True,
        )

        mlflow_artifacts_manager = BlobStorageManager(
            container_name=os.environ["DATA_MLFLOW_SOURCE_CONTAINER_NAME"],
            account_url=os.environ["DATA_MLFLOW_SOURCE_ACCOUNT_URL"],
        )

        mlflow_artifacts_manager.download_folder(
            FOLDER_SCTAB_MLFLOW_PATH, destination_path="./"
        )

        shutil.move(f"./{FOLDER_SCTAB_MLFLOW_PATH}/", DATA_SCTAB_PATH)
        shutil.rmtree("./experiments")

    genes_from_model = pd.read_parquet(VAR_SCTAB_PATH)
    print(genes_from_model)

    set_genes_model = set(genes_from_model["feature_name"].values)
    set_genes_data = set(list(adata_combined.var_names))
    set_intersection = set_genes_data.intersection(set_genes_model)
    intersection = len(set_intersection)
    list_intersection = list(set_intersection)

    percentage_intersection = intersection/len(set_genes_model)

    print('##################################################################')
    print(f'The quantity of genes in this dataset is {len(set_genes_data)}')
    print(f'The quantity of genes in the model is {len(set_genes_model)}')
    print(f'The percentage of intesection for sctab is {percentage_intersection}')
    print('##################################################################')


    adata_combined = adata_combined[
        :, adata_combined.var.index.isin(genes_from_model.feature_name)
    ]
    x_streamlined = streamline_count_matrix(
        csc_matrix(adata_combined.X),
        adata_combined.var.index,  # change this if gene names are stored in different column
        genes_from_model.feature_name,
    )
    loader = dataloader_factory(x_streamlined, batch_size=BATCH_SIZE_SCTAB)

    # load checkpoint
    if torch.cuda.is_available():
        ckpt = torch.load(
            CHECKPOINT_SCTAB_PATH,
        )
    else:
        ckpt = torch.load(
            CHECKPOINT_SCTAB_PATH,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    tabnet_weights = OrderedDict()
    for name, weight in ckpt["state_dict"].items():
        if "classifier." in name:
            tabnet_weights[name.replace("classifier.", "")] = weight

    # load in hparams file of model to get model architecture
    with open(HPARAMS_SCTAB_PATH) as f:
        model_params = yaml.full_load(f.read())

    # initialzie model with hparams from hparams.yaml file
    tabnet = TabNet(
        input_dim=model_params["gene_dim"],
        output_dim=model_params["type_dim"],
        n_d=model_params["n_d"],
        n_a=model_params["n_a"],
        n_steps=model_params["n_steps"],
        gamma=model_params["gamma"],
        n_independent=model_params["n_independent"],
        n_shared=model_params["n_shared"],
        epsilon=model_params["epsilon"],
        virtual_batch_size=model_params["virtual_batch_size"],
        momentum=model_params["momentum"],
        mask_type=model_params["mask_type"],
    )

    # load trained weights
    tabnet.load_state_dict(tabnet_weights)
    # set model to inference mode
    tabnet.eval()

    preds = []

    ### Use this
    with torch.no_grad():
        for batch in tqdm(loader):
            # normalize data
            x_input = sf_log1p_norm(batch[0]["X"])
            logits, _ = tabnet(x_input)
            preds.append(torch.argmax(logits, dim=1).numpy())

    preds = np.hstack(preds)

    cell_type_encoding = pd.read_parquet(CELLTYPE_SCTAB_PATH)
    preds = cell_type_encoding.loc[preds]["label"].to_numpy()

    with open(MAPPING_SCTAB_PATH, "r") as f:
        output_mapping = json.load(f)

    df_predictions = pd.DataFrame(
        {"sample": adata_combined.obs.index, "predicted": preds}
    )
    df_predictions = df_predictions.set_index("sample")
    df_predictions.index.name = None

    df_processed = label_mapping(df_predictions, output_mapping)

    columns_processed = [
        "predicted",
        PREDICTED_MAPPED_SCTAB_COLUMN,
    ]
    columns_scTab = [
        "predicted_scTab",
        PREDICTED_MAPPED_SCTAB_COLUMN,
    ]

    adata_combined.obs[columns_scTab] = df_processed[columns_processed]

    return adata_combined, percentage_intersection, list_intersection


def unified_models_output(adata_combined: ad.AnnData) -> ad.AnnData:
    """
    Create a unified cell type prediction by combining CancerFinder and scTab model outputs.

    This function creates a single column 'metapipeline_automatic_cell_type' that:
    - Uses CancerFinder predictions for cells classified as malignant
    - Uses scTab predictions for cells classified as non-malignant

    The function handles three scenarios:
    1. Both CancerFinder and scTab predictions are available: combines them based on malignancy status
    2. Only CancerFinder predictions are available: uses CancerFinder predictions
    3. Only scTab predictions are available: uses scTab predictions

    Parameters
    ----------
    adata_combined : AnnData
        Combined AnnData object containing CancerFinder and/or scTab predictions in obs

    Returns
    -------
    AnnData
        AnnData object with unified cell type predictions added to obs[MERGED_MODELS_COLUMN]
    """
    condition_cancerfinder = CLASS_CANCERFINDER_COLUMN in adata_combined.obs.columns
    condition_sctab = PREDICTED_MAPPED_SCTAB_COLUMN in adata_combined.obs.columns

    if condition_cancerfinder and condition_sctab:
        # Create a mask for non-malignant cells (where CancerFinder != "Malignant cell")
        non_malignant_mask = (
            adata_combined.obs[CLASS_CANCERFINDER_COLUMN] != CANCERFINDER_1_CLASS
        )

        # Initialize the unified column with CancerFinder predictions
        adata_combined.obs[MERGED_MODELS_COLUMN] = adata_combined.obs[
            CLASS_CANCERFINDER_COLUMN
        ]

        # Replace non-malignant cells with scTab predictions
        adata_combined.obs.loc[non_malignant_mask, MERGED_MODELS_COLUMN] = (
            adata_combined.obs.loc[non_malignant_mask, PREDICTED_MAPPED_SCTAB_COLUMN]
        )
    elif condition_cancerfinder:
        # Only CancerFinder predictions available
        non_malignant_mask = (
            adata_combined.obs[CLASS_CANCERFINDER_COLUMN] != CANCERFINDER_1_CLASS
        )
        adata_combined.obs[MERGED_MODELS_COLUMN] = adata_combined.obs[
            CLASS_CANCERFINDER_COLUMN
        ]
        adata_combined.obs.loc[non_malignant_mask, MERGED_MODELS_COLUMN] = (
            "Not available"
        )

    elif condition_sctab:
        # Only scTab predictions available
        adata_combined.obs[MERGED_MODELS_COLUMN] = adata_combined.obs[
            PREDICTED_MAPPED_SCTAB_COLUMN
        ]
    return adata_combined