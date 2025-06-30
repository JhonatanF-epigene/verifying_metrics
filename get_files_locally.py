import logging
import os
import traceback

from data_integration_utils.azure_blob_manager import (
    ClinicalBronzeBSM,
    OmicBronzeBSM,
)
from dotenv import load_dotenv
import load_gene_standardisation_df
from utils import (
    check_3_files_per_sample_prefix,
    check_file_type,
    check_no_intersection_pattern_list,
    combine_single_cell_data,
    extract_molecular_data,
    filter_raw_count_files,
    get_h5_files_prefix,
    get_tar_path,
    execute_cancer_finder,
    execute_sctab,
    list_files_RAW,
    split_gene_standardisation_by_reference,
    unzip_genes_barcode_matrix_files_triplets,
)
from constants import LogStep


logging.basicConfig(level=logging.INFO)

load_dotenv("env/.env")


def prection_bronze(dataset: str, data_path: str, sc_file_type: str, cancer_finder, scTab) -> None:
    """
    Preprocess single cell data and upload to Azure Blob Storage

    Parameters
    ----------
    dataset : str
        Dataset identifier
    data_path : str
        Path to download the data
    sc_file_type :str
        Single cell file type (TAR_H5, TAR_MTX, TAR_CSV, TAR_TSV, TAR_TXT)

    """
    logging.info("Download of scRNA-Seq data started")
    logging.info(f"Processing:{dataset}")

    assert sc_file_type in ["TAR_H5", "TAR_MTX", "TAR_CSV", "TAR_TSV", "TAR_TXT"], (
        f"File type {sc_file_type} not supported"
    )

    clinical_bronze_manager = ClinicalBronzeBSM()

    context = clinical_bronze_manager.get_context_json(dataset)

    gene_standardisation_df = load_gene_standardisation_df()
    # Can be moved to gene_name_harmonisation
    splitted_reference = split_gene_standardisation_by_reference(
        gene_standardisation_df
    )
    try:
        if len(context["supplementary_file"]) > 0:
            dataset_folder = os.path.join(data_path, dataset)

            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)

            get_molecular_data(
                context_data_sup_file_path_list=context["supplementary_file"],
                work_dir=dataset_folder,
            )

            tar_path = get_tar_path(context["supplementary_file"])
            tar_name = extract_molecular_data(
                sup_file=tar_path, work_dir=dataset_folder
            )

            files_list = list_files_RAW(data_path=dataset_folder, dataset=dataset)

            if sc_file_type == "TAR_H5":
                file_prefixes = get_h5_files_prefix(files_list)
            elif sc_file_type == "TAR_MTX":
                sample_prefix_genes, sample_prefix_feature = check_file_type(files_list)
                if check_no_intersection_pattern_list(
                    sample_prefix_genes, sample_prefix_feature
                ):
                    logging.info("No intersection between gene and feature files")
                else:
                    logging.error("Intersection between gene and feature files")

                file_prefixes = sample_prefix_genes + sample_prefix_feature

                if check_3_files_per_sample_prefix(file_prefixes, files_list):
                    logging.info("3 files per sample prefix")
                else:
                    logging.error("Not 3 files per sample prefix")

                # genes/barcodes/matrix files are unzipped in the work_dir
                if len(sample_prefix_genes) > 0:
                    unzip_genes_barcode_matrix_files_triplets(
                        work_dir=os.path.join(dataset_folder, tar_name),
                        files_list=files_list,
                        pattern_list=sample_prefix_genes,
                    )
            elif sc_file_type in ("TAR_CSV", "TAR_TXT", "TAR_TSV"):
                raw_counts_file_list = filter_raw_count_files(files_list)
                file_prefixes = [
                    file for file in raw_counts_file_list if file.startswith("GSM")
                ]
            else:
                raise ValueError(f"File type {sc_file_type} not supported")

            adata_combined = combine_single_cell_data(
                file_prefixes=file_prefixes,
                data_path=dataset_folder,
                dataset=dataset,
                sc_file_type=sc_file_type,
                splitted_reference=splitted_reference,
            )
            adata_combined.write_zarr(
                os.path.join(dataset_folder, f"{dataset}_raw_data.zarr")
            )

            if cancer_finder:
                execute_cancer_finder(adata_combined, slack_conf)

            if scTab:
                adata_combined = execute_sctab(adata_combined)

            adata_combined.write_zarr(
                os.path.join(dataset_folder, f"{dataset}_raw_data.zarr")
            )

            omic_bronze_manager = OmicBronzeBSM()
            omic_bronze_manager.upload_folder(
                omic_bronze_manager.get_dataset_filename(dataset, "scRNA-Seq"),
                os.path.join(dataset_folder, f"{dataset}_raw_data.zarr"),
                recursive=True,
            )
        else:
            raise ValueError("No supplementary file for this dataset")
    except Exception as e:
        logging.error("".join(traceback.format_tb(e.__traceback__)))
        raise e
