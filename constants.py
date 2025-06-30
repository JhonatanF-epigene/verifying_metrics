"""
Module containing constants used in the single cell project.
"""

from enum import Enum

COUNT_PATTERN = ("count", "counts", "umi", "raw")
UNWANTED_PATTERN = ("normalized", "tcr", "bcr", "contig")

GENE_NAME_COLUMN = "gene_name"
ENTREZ_ID_COLUMN = "entrez_id"
GENE_ID_COLUMN = "gene_id"
GENE_INTER_SIZE = 20000

### CancerFinder variables
DATA_CANCERFINDER_PATH = "./data_CancerFinder"
CHECKPOINT_CANCERFINDER_PATH = f"{DATA_CANCERFINDER_PATH}/sc_pretrain_article.pkl"
CHECKPOINT_CANCERFINDER_AZURE_PATH = (
    "experiments/predict_cancerfinder/sc_pretrain_article.pkl"
)
GENES_CANCERFINDER_PATH = f"{DATA_CANCERFINDER_PATH}/cancer_finder_gene_list.json"
GENES_CANCERFINDER_AZURE_PATH = (
    "experiments/predict_cancerfinder/cancer_finder_gene_list.json"
)
OUTPUT_CANCERFINDER_COLUMN = "predict_CancerFinder"
CLASS_CANCERFINDER_COLUMN = "prediction_CancerFinder"
CANCERFINDER_1_CLASS = "Malignant cell"
CANCERFINDER_0_CLASS = "Non malignant"

MLFLOW_SOURCE_CONTAINER_NAME_VARIABLE = "DATA_MLFLOW_SOURCE_CONTAINER_NAME"
MLFLOW_SOURCE_ACCOUNT_URL_VARIABLE = "DATA_MLFLOW_SOURCE_ACCOUNT_URL"
PREDICTED_MAPPED_SCTAB_COLUMN = "predicted_mapped_scTab"
PREDICTED_MAPPED_OG_SCTAB_COLUMN = "predicted_mapped_og_scTab"

### scTab variables related
PREDICTED_MAPPED_SCTAB_COLUMN = "predicted_mapped_scTab"
DATA_SCTAB_PATH = "./data_scTab"
CHECKPOINT_SCTAB_PATH = (
    f"{DATA_SCTAB_PATH}/scTab-checkpoints/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt"
)
FOLDER_SCTAB_MLFLOW_PATH = "experiments/predict_scTab/data_scTab"
VAR_SCTAB_PATH = f"{DATA_SCTAB_PATH}/var.parquet"
HPARAMS_SCTAB_PATH = f"{DATA_SCTAB_PATH}/scTab-checkpoints/hparams.yaml"
MAPPING_SCTAB_PATH = f"{DATA_SCTAB_PATH}/output_mapping.json"
CELLTYPE_SCTAB_PATH = f"{DATA_SCTAB_PATH}/cell_type.parquet"
BATCH_SIZE_SCTAB = 2048

MERGED_MODELS_COLUMN = "metapipeline_automatic_cell_type"

# Logging constants
class LogStep(Enum):
    PREPROCESSING = "pre-processing"
    NORMALIZATION = "normalization"
    GENE_NAME_HARMONIZATION = "gene name harmonization"