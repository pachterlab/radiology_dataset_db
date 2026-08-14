import json
import logging
import os
import subprocess

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

#$ for real time, set to None
IDS_TO_KEEP = None
# IDS_TO_KEEP = {
#     "36204533",  # RadImageNet
#     "31831740",  # MIMIC-CXR
#     "32457287",  # UK Biobank
#     "23884657",  # TCIA
#     "41781626",  # Merlin
# }

def get_model() -> str:
    model = os.getenv("MODEL", "openai:Qwen/Qwen2.5-7B-Instruct")
    vllm_port = os.getenv("VLLM_PORT")
    if vllm_port is not None:
        vllm_port = int(vllm_port)

        result = subprocess.run(["curl", f"http://localhost:{vllm_port}/v1/models"], capture_output=True, text=True)
        if result.returncode == 0:
            result = json.loads(result.stdout)
            model_id = result["data"][0]["id"]
            model = f"openai:{model_id}"
        else:
            print("VLLM request failed - check if it is running and the port is correct.")
            print("stderr:", result.stderr)
    return model

MODEL = get_model()

#* is_database_paper_classifier_llm.py
CLASSIFICATION_INSTRUCTIONS = (
    "Determine whether the paper INTRODUCES or CREATES a dataset.\n"
    "Return is_dataset_creation = true if:\n"
    "- The paper develops, constructs, introduces, or presents a dataset/database/benchmark\n"
    "- Even if the dataset has no explicit name\n\n"
    "Return false if:\n"
    "- The paper only uses existing datasets\n"
    "- It is a methods/model paper\n"
    "- It analyzes data without creating a dataset\n\n"
    "Be conservative: if unsure, return true."
)

CLASSIFICATION_AGENT_INSTRUCTIONS = "Classify whether this paper creates a dataset"

#* check_if_dataset_is_available_llm.py
DATASET_AVAILABILITY_INSTRUCTIONS = (
    "Determine whether the paper indicates that its dataset is publicly available.\n"
    "Return is_publicly_available = true if there is direct evidence in the provided text such as:\n"
    "- explicit language like 'open', 'publicly available', 'public', 'available', 'release', 'accessible', 'open-access', or a data availability statement\n"
    "- a non-DOI URL (e.g., GitHub, Zenodo, Kaggle, institutional repository, challenge site, or dataset website)\n"
    "- wording that readers can access or download the dataset\n\n"
    "Return false if:\n"
    "- the text only says data are available on request\n"
    "- access is restricted, private, proprietary, or behind an application process\n"
    "- only a DOI to the paper is provided without dataset access evidence\n"
    "- there is no clear evidence of public dataset availability\n\n"
    "Use title, abstract, and full text (if provided). Be generous, especially if full text is not provided: if unsure, return true."
)

DATASET_AVAILABILITY_AGENT_INSTRUCTIONS = (
    "Classify whether the dataset described in this paper is publicly available."
)

#* query_synthesis_llm.py
# Shared rules about how a PubMed search term must be written. Every term-proposing
# agent gets these, because ungrounded/malformed terms are the main failure mode.
QUERY_TERM_FORMAT_RULES = """
Every term you propose MUST be a single valid PubMed search term with a field tag:
- MeSH descriptor:  "Radiography"[MeSH]  or  "Databases, Factual"[MeSH]
- Title only:       dataset[ti]
- Title/abstract:   radiograph[tiab]  or  "chest radiograph"[tiab]
Rules:
- Quote any term containing a space. Never quote a single word.
- Only use [MeSH] for strings you are confident are real MeSH descriptors. When unsure, use [tiab].
- Truncation with * is allowed, e.g. crystallograph*[tiab].
- Propose ONE concept per term. Never put OR, AND, NOT, or parentheses inside a term.
- Do not include the field tag twice and do not wrap terms in extra brackets.
"""

SEED_VERIFICATION_INSTRUCTIONS = (
    "You are validating a paper that a user supplied as a required example for a literature search.\n"
    "Given the user's topic description and the paper's title and abstract, decide two things:\n"
    "1. is_on_topic: does this paper fall within the user's stated topic?\n"
    "2. presents_dataset: does this paper INTRODUCE, CREATE, or RELEASE a dataset, database, "
    "cohort, or benchmark? Papers that merely reuse existing data are false.\n"
    "Give a one-sentence reason. Be strict: the user is asserting this paper must appear in "
    "the search results, so a bad example will distort the entire query."
)

SEED_VERIFICATION_AGENT_INSTRUCTIONS = "Verify this seed paper against the topic"

QUERY_PROPOSAL_INSTRUCTIONS = (
    "You design PubMed boolean queries that find papers introducing datasets on a given topic.\n"
    "Produce three groups of terms which will be assembled as:\n"
    "    (dataset_terms) AND (topic_terms) NOT (exclude_terms)\n\n"
    "- dataset_terms: signals that a paper INTRODUCES a dataset/database/benchmark/cohort. "
    "These are mostly topic-independent, e.g. dataset[ti], database[ti], benchmark[ti], "
    "\"data repository\"[ti], \"Databases, Factual\"[MeSH]. Prefer [ti] over [tiab] here: a paper "
    "that introduces a dataset usually says so in its title, while [tiab] pulls in every paper "
    "that merely mentions using a dataset.\n"
    "- topic_terms: subject matter of the user's topic, including MeSH descriptors, synonyms, "
    "abbreviations, and common acronyms. Be generous here; recall matters most on this axis.\n"
    "- exclude_terms: terms that catch known false-positive traps for this topic, e.g. an "
    "acronym that means something different in another field. Leave empty if none are obvious. "
    "Be very conservative: an exclusion that is too broad silently destroys recall.\n\n"
    "Aim for 8-15 dataset_terms and 10-25 topic_terms.\n"
    + QUERY_TERM_FORMAT_RULES
)

QUERY_PROPOSAL_AGENT_INSTRUCTIONS = "Propose PubMed query terms for this topic"

QUERY_RECOVERY_INSTRUCTIONS = (
    "A PubMed query is failing to retrieve a paper that MUST be in the results.\n"
    "You are shown the query block that the paper fails to match, and the paper itself "
    "(including its real MeSH terms as indexed by PubMed).\n"
    "Propose 1-4 additional terms for that block which would make the paper match.\n"
    "Prefer terms taken verbatim from the paper's own indexed MeSH terms or its title, since "
    "those are guaranteed to match. Do NOT propose a term so specific that it only ever matches "
    "this one paper (e.g. the dataset's proper name) unless nothing else can work -- the query "
    "must still generalize to papers nobody has told you about.\n"
    + QUERY_TERM_FORMAT_RULES
)

QUERY_RECOVERY_AGENT_INSTRUCTIONS = "Propose terms to recover this missing paper"

QUERY_EXCLUSION_INSTRUCTIONS = (
    "A PubMed query is retrieving a paper that the user says must NOT appear in the results.\n"
    "Propose 1-3 exclusion terms that would remove it. The terms are placed in a NOT block, so "
    "anything matching them is dropped from the results entirely.\n"
    "Choose terms that characterize what makes this paper wrong for the topic, not terms that "
    "merely happen to appear in it. A term that also appears in on-topic papers will silently "
    "destroy recall, so prefer precise, narrow terms and propose nothing if you cannot find a "
    "safe discriminator.\n"
    + QUERY_TERM_FORMAT_RULES
)

QUERY_EXCLUSION_AGENT_INSTRUCTIONS = "Propose exclusion terms to remove this paper"

QUERY_BROADENING_INSTRUCTIONS = (
    "A PubMed query is retrieving too few results to plausibly cover its topic.\n"
    "Propose 3-8 additional topic terms that would broaden recall: synonyms, alternative "
    "spellings, acronyms, related MeSH descriptors, and adjacent subject vocabulary.\n"
    "Do not propose terms already present in the query, and do not drift off topic.\n"
    + QUERY_TERM_FORMAT_RULES
)

QUERY_BROADENING_AGENT_INSTRUCTIONS = "Propose additional topic terms to broaden this query"

TOPIC_RELEVANCE_INSTRUCTIONS = (
    "Decide whether a paper falls within the user's stated topic, based on its title and abstract.\n"
    "Judge topical fit only; do not judge whether the paper introduces a dataset.\n"
    "Be strict: this judgement is used to estimate the precision of a literature search."
)

TOPIC_RELEVANCE_AGENT_INSTRUCTIONS = "Judge whether this paper is on topic"

#* extract_radiology_dataset_information_llm.py
# MeSH terms: https://www.ncbi.nlm.nih.gov/mesh/?term=%22radiology%22%5BMeSH%20Terms%5D%20OR%20%22radiographic%22%5BMeSH%20Terms%5D%20OR%20%22radiography%22%5BMeSH%20Terms%5D%20OR%20radiology%5BText%20Word%5D&cmd=DetailsSearch
PUBMED_QUERY_RADIOLOGY = """
("Database Management Systems"[MeSH] OR dataset[ti] OR database[ti] OR "data collection"[ti] OR "information repository"[ti] OR benchmark[ti] OR "challenge data"[ti] OR "data commons"[ti] OR "data repository"[ti] OR "data sharing"[ti])
AND ("Radiology"[MeSH] OR "Radiography"[MeSH] OR "Radiology Information Systems"[MeSH] OR radiology[tiab] OR radiograph[tiab] OR radiographs[tiab] OR "Diagnostic Imaging"[tiab] OR "Medical Image"[tiab] OR "Medical Imaging"[tiab] OR "Biomedical Image"[tiab] OR "Biomedical Imaging"[tiab] OR XR[tiab] OR CT[tiab] OR MRI[tiab] OR PET[tiab] OR SPECT[tiab] OR "X-ray"[tiab] OR "Computed Tomography"[tiab] OR "Magnetic Resonance"[tiab] OR Ultrasound[tiab] OR "Positron Emission Tomography"[tiab] OR "Single Photon Emission Computed Tomography"[tiab])
NOT (("Nuclear Magnetic Resonance"[tiab] OR NMR[tiab]) OR ("X-ray crystallography"[tiab] OR crystallograph*[tiab] OR diffraction[tiab]) OR (mice[tiab] or mouse[tiab]))
"""  # removed "Databases, Factual"[MeSH] because it dropped search space from 12319 to 3877 while keeping all of my test cases

EXTRACTION_INSTRUCTIONS_RADIOLOGY = (
    "You MUST extract a dataset/benchmark name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "Examples:\n"
    "- UK Biobank\n"
    "- MIMIC-CXR\n"
    "- RadImageNet\n"
    "- LiTS\n"
)

ADD_TEXT_EXTRACT_RADIOLOGY = f"""
Extract:
- dataset name (best candidate; often found directly in the title, e.g. "UK Biobank", "MIMIC-CXR", "RadImageNet" — use the name as it appears)
- number of patients or participants
- number of imaging studies or images
- imaging modalities: CT, MRI, X-ray, US, PET
- body regions: brain, chest, abdomen, pelvis, limbs
- additional data (reports, captions, segmentation, genomics, VQA, etc)
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_RADIOLOGY = "Extract dataset information"

#* extract_scrnaseq_dataset_information_llm.py
PUBMED_QUERY_SCRNASEQ = ""

EXTRACTION_INSTRUCTIONS_SCRNASEQ = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for scRNA-seq/snRNA-seq dataset papers.\n"
)

ADD_TEXT_EXTRACT_SCRNASEQ = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- sequencing technology/platform (e.g., 10X, SMART-Seq/SMARTSEQ, Parse, Drop-seq, inDrops, Seq-Well)
- disease(s)
- species
- tissue(s)
- whether the assay is cell, nuclei, or both (for scRNA-seq/snRNA-seq)
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_SCRNASEQ = "Extract scRNA-seq/snRNA-seq dataset information"

#* extract_bulk_genomics_dataset_information_llm.py
PUBMED_QUERY_BULK_GENOMICS = ""

EXTRACTION_INSTRUCTIONS_BULK_GENOMICS = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for bulk genomics dataset papers.\n"
)

ADD_TEXT_EXTRACT_BULK_GENOMICS = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- sequencing modality/modality labels from: WGS, WXS, bulk RNA
- whether raw reads are available (true/false if evidence is present)
- disease(s)
- species
- tissue(s)
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_BULK_GENOMICS = "Extract bulk genomics dataset information"

#* extract_spatial_transcriptomics_dataset_information_llm.py
PUBMED_QUERY_SPATIAL_TRANSCRIPTOMICS = ""

EXTRACTION_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS = (
    "You MUST extract a dataset name.\n"
    "Never return null for name.\n"
    "If uncertain, choose the most likely dataset or cohort name.\n"
    "Prefer names in the title.\n"
    "This extractor is for spatial transcriptomics dataset papers.\n"
)

ADD_TEXT_EXTRACT_SPATIAL_TRANSCRIPTOMICS = f"""
Extract:
- dataset name (best candidate; often found directly in the title)
- number of patients or participants
- spatial transcriptomics technology/platform (e.g., Visium, Visium HD, Xenium, CosMx, MERFISH, seqFISH, Stereo-seq, Slide-seq, GeoMx DSP)
- disease(s)
- species
- tissue(s)
- whether the assay is cell, nuclei, or both
- the http link to the dataset, if available at the end of the abstract
"""

EXTRACTION_AGENT_INSTRUCTIONS_SPATIAL_TRANSCRIPTOMICS = "Extract spatial transcriptomics dataset information"

#* add additional instructions and config variables for other modalities here, e.g. genomics, pathology, etc

PUBMED_QUERY_DICT = {
    "radiology": PUBMED_QUERY_RADIOLOGY,
    "scrnaseq": PUBMED_QUERY_SCRNASEQ,
    "bulk_genomics": PUBMED_QUERY_BULK_GENOMICS,
    "spatial_transcriptomics": PUBMED_QUERY_SPATIAL_TRANSCRIPTOMICS,
}
SUPPORTED_MODALITIES = set(PUBMED_QUERY_DICT.keys())
#* add additional PubMed queries for other modalities here, e.g. genomics, pathology, etc