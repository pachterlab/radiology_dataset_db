import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
from typing import List

import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from radiology_dataset_db.config import IDS_TO_KEEP, LOG_LEVEL, MODEL, SUPPORTED_MODALITIES, PUBMED_QUERY_DICT
#* add additional extraction instructions and functions for other modalities here, e.g. genomics, pathology, etc
from radiology_dataset_db import extract_bulk_genomics_dataset_info_with_agent, extract_radiology_dataset_info_with_agent, extract_scrnaseq_dataset_info_with_agent, extract_spatial_transcriptomics_dataset_info_with_agent
from radiology_dataset_db.check_if_dataset_is_available_llm import check_if_dataset_is_available
from radiology_dataset_db.pubmed_utils import (
    add_column_to_isolate_mesh_terms_from_pubmed_matches,
    extract_pubmed_metadata, fetch_pubmed_citation_counts,
    fetch_pubmed_details, search_pubmed, try_to_get_full_text, check_url)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Build database from PubMed and LLM extraction.")

    parser.add_argument("-m", "--database-modality", type=str, default="radiology", choices=SUPPORTED_MODALITIES, help=f"Modality of datasets to extract. Supported modalities: {SUPPORTED_MODALITIES}.")
    parser.add_argument("--topic-spec", type=str, default=None, help="Path to a topic spec JSON written by scripts/build_query.py. Its query replaces the built-in query for --database-modality (which then only selects the extraction schema).")
    parser.add_argument("-o", "--output-path", type=str, default=None, help="Path to save the output CSV file. Defaults to data/{database_modality}_db.csv.")
    parser.add_argument("--output-path-failed", type=str, default=None, help="Path to save the CSV file containing metadata of articles for which dataset extraction failed.")
    parser.add_argument("-max", "--max-papers", type=int, default=9999, help="Maximum number of papers to retrieve from PubMed for processing.")
    parser.add_argument("-min", "--min-citations", type=int, default=25, help="Minimum number of citations for a paper to be included.")
    parser.add_argument("--citation-number-grace-period-years", type=int, default=1, help="Number of years to allow for citation count grace period.")  #  e.g., if set to 1 and the current year is 2026, then papers published in 2025 and 2026 will be exempt from the citation filter - set to 0 to disable the grace period
    parser.add_argument("--num-tries-agent", type=int, default=7, help="Number of times to retry dataset extraction for each paper.")
    parser.add_argument("--article-timeout-seconds", type=float, default=300.0, help="Max seconds allowed for processing a single article before marking it as failed. Set <=0 to disable timeout.")
    parser.add_argument("--disable-check-dataset-is-available", action="store_true", help="Disable LLM availability check. By default, availability is checked and written to output as dataset_is_available.")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads to use for parallel processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = f"data/{args.database_modality}_db.csv"
    return args

def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set.")
    return value


async def extract_dataset_with_retries(
    title: str,
    abstract: str,
    publication_metadata: dict,
    database_modality: str,
    num_tries_agent: int,
):
    dataset = None
    for _ in range(num_tries_agent):
        if database_modality == "radiology":
            dataset = await extract_radiology_dataset_info_with_agent(title, abstract, publication_metadata)
        elif database_modality == "scrnaseq":
            dataset = await extract_scrnaseq_dataset_info_with_agent(title, abstract, publication_metadata)
        elif database_modality == "bulk_genomics":
            dataset = await extract_bulk_genomics_dataset_info_with_agent(title, abstract, publication_metadata)
        elif database_modality == "spatial_transcriptomics":
            dataset = await extract_spatial_transcriptomics_dataset_info_with_agent(title, abstract, publication_metadata)
        #* add additional modalities here with corresponding extraction functions, e.g. genomics, pathology, etc
        else:
            raise ValueError(
                f"Unsupported database modality: {database_modality}. Supported modalities: {SUPPORTED_MODALITIES}."
            )
        if dataset is not None:
            break
    return dataset


async def process_publication_metadata(
    publication_metadata: dict,
    check_dataset_is_available: bool,
    database_modality: str,
    num_tries_agent: int,
    timeout_seconds: float | None,
):
    title = publication_metadata.get("title")
    abstract = publication_metadata.get("abstract")
    if timeout_seconds is not None and timeout_seconds > 0:
        dataset = await asyncio.wait_for(
            extract_dataset_with_retries(
                title,
                abstract,
                publication_metadata,
                database_modality,
                num_tries_agent,
            ),
            timeout=timeout_seconds,
        )
    else:
        dataset = await extract_dataset_with_retries(
            title,
            abstract,
            publication_metadata,
            database_modality,
            num_tries_agent,
        )

    if isinstance(dataset, BaseModel):
        logger.debug(f"Extraction successful for article: {title}.")
        dataset_is_available = None
        if check_dataset_is_available:
            full_text = None  # try_to_get_full_text(article)  # TODO: (1) write a function to extract just part of full text (eg just intro, methods, and data/code availability) and convert from byte --> str, and (2) determine what to do if full text not available (perhaps skip entirely?)
            dataset_is_available = await check_if_dataset_is_available(title, abstract, full_text=full_text)
            logger.debug(f"Dataset availability check complete for article: {title}. dataset_is_available={dataset_is_available}")
        return dataset, None, dataset_is_available

    logger.debug(f"Extraction failed for article: {title}")
    return None, publication_metadata, None


def process_article_threaded(
    article,
    citation_counts_by_pmid,
    titles_existing,
    check_dataset_is_available,
    database_modality,
    num_tries_agent,
    timeout_seconds,
    pubmed_query
):
    publication_metadata = extract_pubmed_metadata(article, citation_counts_by_pmid, pubmed_query=pubmed_query)
    title = publication_metadata.get("title")

    if title and title in titles_existing:
        logger.debug(f"Article title already exists in output, skipping: {title}")
        return None, None

    try:
        return asyncio.run(
            process_publication_metadata(
                publication_metadata,
                check_dataset_is_available,
                database_modality,
                num_tries_agent,
                timeout_seconds,
            )
        )
    except TimeoutError:
        logger.warning(f"Timed out processing article with title: {title} after {timeout_seconds}s")
        failed = dict(publication_metadata)
        failed["failure_reason"] = f"timeout_after_{timeout_seconds}_seconds"
        return None, failed, None
    except Exception as e:
        logger.error(f"Error processing article with title: {title}. Error: {e}")
        return None, publication_metadata, None

# de-duplicate by keeping oldest (identical title first, then identical name)
def drop_duplicates(df, subset_col, sort_col="paper_year", ascending=True, df_removed=None):
    # df = df.copy()
    df_sorted = df.sort_values(sort_col, ascending=ascending)
    dup_mask = df_sorted.duplicated(subset=[subset_col], keep="first")
    df_kept = df_sorted[~dup_mask]
    df_removed_tmp = df_sorted[dup_mask]
    if df_removed is not None:
        # df_removed = df_removed.copy()
        df_removed = pd.concat([df_removed, df_removed_tmp], ignore_index=True)  # concatenate
    else:
        df_removed = df_removed_tmp
    
    logger.info(f"After de-duplication by {subset_col}, {len(df_kept)} unique datasets remain in the output.")

    return df_kept, df_removed


def save_progress(args, df_existing, extracted_datasets: List[dict], failed_metadata: list, total_articles: int):
    # extracted_datasets is already a list of row dicts
    rows = extracted_datasets
    new_df = pd.DataFrame(rows)

    logger.info(f"Extracted {len(new_df)} datasets from {total_articles} articles processed/fetched.")
    logger.info(f"Failed to extract datasets from {len(failed_metadata)} articles.")

    if new_df.empty and (df_existing is None or df_existing.empty):
        logger.info("No new or existing datasets to write; skipping output CSV save.")
        if args.output_path_failed and args.output_path_failed != "None" and len(failed_metadata) > 0:
            df_failed = pd.DataFrame(failed_metadata)
            df_failed.to_csv(args.output_path_failed, index=False)
            logger.info(f"Failed extraction metadata saved to {args.output_path_failed}")
        return

    # add column to isolate mesh terms in pubmed query
    if "pubmed_matches" in new_df.columns:
        new_df = add_column_to_isolate_mesh_terms_from_pubmed_matches(new_df, source_column="pubmed_matches")

    # check if link works
    if "dataset_link" in new_df.columns:
        new_df["link_works"] = new_df["dataset_link"].apply(check_url, timeout=120)
        link_works_col = new_df.pop("link_works")  # place this column right after dataset_link column
        new_df.insert(new_df.columns.get_loc("dataset_link") + 1, "link_works", link_works_col)

    if not new_df.empty:
        new_df["date_added"] = pd.Timestamp.now().isoformat()

        # convert list fields to comma-separated strings for CSV output
        cols_to_skip_joining = {"pubmed_matches"}
        for col in new_df.columns:
            if not new_df[col].empty and isinstance(new_df[col].iloc[0], list) and col not in cols_to_skip_joining:
                new_df[col] = new_df[col].apply(lambda x: ",".join(x) if isinstance(x, list) else x)

    if df_existing is None or df_existing.empty:
        df = new_df
    else:
        if not new_df.empty:
            logger.info(f"Existing output file has {len(df_existing)} datasets. Combining with {len(new_df)} newly extracted datasets.")
            df = pd.concat([df_existing, new_df], ignore_index=True)
            logger.info(f"Combined with existing data, there are now {len(df)} total datasets.")
        else:
            df = df_existing

    if "paper_title" in df.columns:
        df, _ = drop_duplicates(df, subset_col="paper_title", sort_col="paper_year")

    if "name" in df.columns:
        df, df_removed = drop_duplicates(df, subset_col="name", sort_col="paper_citation_count", ascending=False, df_removed=None)  # sort_col="paper_year", ascending=True
        if len(df_removed) > 0:
            removed_dict = df_removed.to_dict(orient="records")
            for entry in removed_dict:
                failed_metadata.append({k: v for k, v in entry.items()})

    logger.info(f"Failed to extract datasets from {len(failed_metadata)} articles.")

    # Save to CSV
    df.to_csv(args.output_path, index=False)

    if args.output_path_failed and args.output_path_failed != "None":  # catch "None" string from env var
        if len(failed_metadata) == 0:
            logger.info("No failed extractions to save.")
        else:
            df_failed = pd.DataFrame(failed_metadata)
            df_failed.to_csv(args.output_path_failed, index=False)
            logger.info(f"Failed extraction metadata saved to {args.output_path_failed}")

    logger.info(f"Extraction complete. Results saved to {args.output_path}")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
async def main():
    args = parse_args()

    if args.topic_spec:
        with open(args.topic_spec) as handle:
            topic_spec = json.load(handle)
        pubmed_query = " ".join((topic_spec.get("query") or "").split())
        if not pubmed_query:
            raise ValueError(f"Topic spec {args.topic_spec} does not contain a 'query' field.")
        logger.info(f"Using synthesized query from {args.topic_spec} (topic: {topic_spec.get('topic')!r})")
        logger.info(f"Extraction schema comes from --database-modality {args.database_modality}.")
    else:
        pubmed_query = PUBMED_QUERY_DICT.get(args.database_modality)
        pubmed_query = " ".join((pubmed_query or "").split())
        if not pubmed_query:
            raise ValueError(f"No PubMed query found for modality {args.database_modality}. Please check the config or specify a supported modality, or supply one with --topic-spec (see scripts/build_query.py). Supported modalities: {SUPPORTED_MODALITIES}.")

    if os.path.exists(args.output_path):
        if args.overwrite:
            confirm = input(f"Output file {args.output_path} already exists. Do you want to overwrite it? (y/n): ")
            if confirm.lower() != "y":
                logger.info("Exiting without overwriting. Please set overwrite=False to append to the existing file, or move/rename it before running the extraction.")
                return
            os.remove(args.output_path)
            os.remove(args.output_path_failed) if args.output_path_failed and os.path.exists(args.output_path_failed) else None
        else:
            logger.info(f"Output file {args.output_path} already exists. Will write on top of it. To overwrite instead, set overwrite=True or remove the existing file before running.")
    else:
        logger.info(f"No existing output file found at {args.output_path}. A new file will be created.")

    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    logger.info(f"Using model: {MODEL}")

    Entrez.email = getenv("ENTREZ_EMAIL")
    if os.getenv("ENTREZ_API_KEY"):
        Entrez.api_key = getenv("ENTREZ_API_KEY")
        logger.info("Using Entrez API key")

    df = None
    titles_existing, pmids_existing = set(), set()
    if os.path.exists(args.output_path):
        df = pd.read_csv(args.output_path)
        titles_existing = set(df["paper_title"].dropna().tolist())
        pmids_existing = set(df["pmid"].dropna().astype(str))
        logger.info(f"Found {len(pmids_existing)} existing PMIDs in output file. These will be skipped in the new extraction.")

    ids = search_pubmed(pubmed_query, args.max_papers)
    if not ids:
        logger.warning("No articles found.")
        return
    
    logger.info(f"Found {len(ids)} articles from PubMed search.")
    
    if IDS_TO_KEEP:
        ids = [id for id in ids if id in IDS_TO_KEEP]
        logger.info(f"Filtered to {len(ids)} articles based on IDS_TO_KEEP list.")
    
    if pmids_existing:
        ids = [id for id in ids if id not in pmids_existing]
        logger.info(f"After filtering out existing PMIDs, {len(ids)} new articles remain for extraction.")
    
    if args.citation_number_grace_period_years is None:
        args.citation_number_grace_period_years = 0
    get_year = math.floor(args.citation_number_grace_period_years) > 0
    if get_year:
        citation_counts_by_pmid, year_results = fetch_pubmed_citation_counts(ids, get_year=True)
    else:
        citation_counts_by_pmid = fetch_pubmed_citation_counts(ids, get_year=False)
        year_results = {}

    if args.min_citations > 0:
        current_year = pd.Timestamp.now().year
        threshold_year = current_year - math.floor(args.citation_number_grace_period_years)
        
        filtered_ids = []
        for id in ids:
            year = year_results.get(id)
            if year is None:
                filtered_ids.append(id)  # if we can't get the year, be generous and keep the paper for extraction, since we can't apply the citation filter without the year
                continue

            citations = citation_counts_by_pmid.get(id, 0)

            # Grace period logic
            if year >= threshold_year:
                filtered_ids.append(id)
            else:
                if citations >= args.min_citations:
                    filtered_ids.append(id)
        
        ids = filtered_ids
        logger.info(f"After filtering out articles with fewer than {args.min_citations} citations, {len(ids)} articles remain.")
    
    logger.info(f"{len(ids)} articles will be processed for dataset extraction.")
    articles = fetch_pubmed_details(ids)

    if not articles:
        logger.warning("No articles found.")
        return

    check_dataset_is_available = not args.disable_check_dataset_is_available
    if check_dataset_is_available:
        logger.info("Dataset availability check enabled. Results will be written to dataset_is_available column.")
    else:
        logger.info("Dataset availability check disabled.")
    timeout_seconds = args.article_timeout_seconds
    extracted_datasets: List[dict] = []
    failed_metadata = []
    interrupted = False
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if args.article_timeout_seconds is not None and args.article_timeout_seconds <= 0:
        logger.info("Per-article timeout disabled.")

    try:
        if args.threads == 1:
            for article in tqdm(articles):
                publication_metadata = {}
                title = None
                try:
                    publication_metadata = extract_pubmed_metadata(article, citation_counts_by_pmid, pubmed_query=pubmed_query)
                    title = publication_metadata.get("title")

                    if title and title in titles_existing:
                        logger.debug(f"Article title already exists in output, skipping: {title}")
                        continue

                    dataset, failure, dataset_is_available = await process_publication_metadata(
                        publication_metadata,
                        check_dataset_is_available,
                        args.database_modality,
                        args.num_tries_agent,
                        timeout_seconds,
                    )
                    if dataset is not None:
                        row = dataset.model_dump()
                        if check_dataset_is_available:
                            row["dataset_is_available"] = dataset_is_available
                        extracted_datasets.append(row)
                    elif failure is not None:
                        failed_metadata.append(failure)

                    await asyncio.sleep(1)  # rate limit for single-threaded mode

                except KeyboardInterrupt:
                    interrupted = True
                    logger.warning("Interrupted during processing loop. Saving partial progress...")
                    break
                except TimeoutError:
                    logger.warning(f"Timed out processing article with title: {title} after {timeout_seconds}s")
                    failed = dict(publication_metadata)
                    failed["failure_reason"] = f"timeout_after_{timeout_seconds}_seconds"
                    failed_metadata.append(failed)
                    continue
                except Exception as e:
                    logger.error(f"Error processing article with title: {title}. Error: {e}")
                    failed_metadata.append(publication_metadata)
                    continue
        else:
            logger.info(f"Using multithreaded extraction with {args.threads} threads.")
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=args.threads)
            tasks = []
            try:
                tasks = [
                    loop.run_in_executor(
                        executor,
                        process_article_threaded,
                        article,
                        citation_counts_by_pmid,
                        titles_existing,
                        check_dataset_is_available,
                        args.database_modality,
                        args.num_tries_agent,
                        timeout_seconds,
                        pubmed_query
                    )
                    for article in articles
                ]
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    dataset, failure, dataset_is_available = await task
                    if dataset is not None:
                        row = dataset.model_dump()
                        if check_dataset_is_available:
                            row["dataset_is_available"] = dataset_is_available
                        extracted_datasets.append(row)
                    elif failure is not None:
                        failed_metadata.append(failure)
            except KeyboardInterrupt:
                interrupted = True
                logger.warning("Interrupted during threaded processing. Cancelling pending work and saving partial progress...")
                for task in tasks:
                    task.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        logger.warning("Interrupted while awaiting main task. Saving partial progress...")
    finally:
        save_progress(args, df, extracted_datasets, failed_metadata, total_articles=len(articles))
        if interrupted:
            logger.warning("Partial progress has been saved after interrupt.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt at process level; exiting.")
