# Radiology Dataset Database

A pipeline for automatically discovering, extracting, and structuring radiology datasets from the literature.

- `data/`: contains the final dataset table (e.g., `radiology_db.csv`)
- `notebooks/`: tutorials and exploratory data analysis notebooks
- `scripts/`: scripts for running the database building pipeline
- `radiology_dataset_db/`: source code for querying PubMed, extracting dataset metadata, and building the database
- `tests/`: pytest-based testing suite

## ⚙️ Installation

### 1. Install and run vLLM (for local LLM inference)

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 2. Set up the radiology dataset database environment

```bash
git clone git@github.com:pachterlab/radiology_dataset_db.git
cd radiology_dataset_db
conda create -n radiology_dataset_db python=3.10 -y
conda activate radiology_dataset_db
pip install -e .
```

### 3. Rename `.env_sample` to `.env` and fill in your Entrez email and API key (optional but recommended for higher rate limits)

## 🚀 Usage
Modify `.env` and `radiology_dataset_db/config.py` as needed to customize PubMed query/LLM settings. Runtime defaults like modality/output paths are now configured via CLI args in `scripts/build_db.py`. Then run:
```bash
python scripts/build_db.py --database-modality MODALITY
```

## 🔎 Synthesizing a PubMed query for a new topic

Writing the PubMed query is the step that previously required a human (see
`notebooks/pubmed_search.ipynb` for the manual drop-one workflow). `scripts/build_query.py`
does that search automatically: you describe the topic in plain language and, optionally,
name a few papers that must (or must not) show up.

```bash
python scripts/build_query.py \
  --topic "chest x-ray datasets" \
  --must-include 31831740 10.1148/ryai.210315 https://pubmed.ncbi.nlm.nih.gov/36204533/ \
  --must-exclude 12345678 \
  --max-results 8000 \
  --audit-sample 20
```

`--must-include` / `--must-exclude` accept PMIDs, DOIs, PubMed URLs, doi.org URLs, and PMCIDs.

What it does:

1. **Resolves and verifies your seeds.** Each must-include paper is fetched from PubMed and
   checked by an LLM: is it on topic, and does it actually introduce a dataset? Papers that
   fail are reported and dropped from the anchor set, so one bad example cannot drag the
   whole query off topic (`--keep-rejected-seeds` overrides this, `--no-verify-seeds` skips it).
2. **Proposes a query** of the form `(dataset terms) AND (topic terms) NOT (exclusions)`.
   Every proposed `[MeSH]` term is checked against the real MeSH vocabulary; hallucinated
   descriptors are downgraded to `[tiab]` rather than silently matching nothing.
3. **Hillclimbs against live PubMed.** Each iteration measures the query and applies one
   repair: recover a missing must-include paper (diagnosing *which* block excluded it),
   exclude a must-exclude paper, prune a term that costs many hits but no required paper,
   or broaden a query that is implausibly narrow. The LLM proposes vocabulary; PubMed
   decides. An edit is kept only if the measured counts and seed membership improve, and
   a term is never dropped if a must-include paper depends on it.
4. **Optionally audits precision** (`--audit-sample N`): samples N results with a fixed
   random seed and has the LLM judge how many are genuinely on-topic dataset papers. Hit
   count tells you whether a query is big; this tells you whether it is right.

The result is written to `topics/{slug}.json` — the query, the term-by-term hit
contributions, the full iteration history, and the seed verdicts — and can be fed straight
into the extraction pipeline:

```bash
python scripts/build_db.py --topic-spec topics/chest_x_ray_datasets.json --database-modality radiology
```

`--topic-spec` supplies the query; `--database-modality` still selects the extraction schema.

To refine an existing hand-written query instead of starting from scratch:

```bash
python scripts/build_query.py --topic "radiology datasets" --start-modality radiology
```

Note that `--max-results` is a budget, not a guarantee: if the only way to fit it is to drop
a term a must-include paper depends on, the agent keeps the paper and reports the overage.
Seed recall is the constraint; result count is the objective.

## Currently supported modalities:
- Radiology: `--database-modality radiology`
- Single-cell RNA-seq: `--database-modality scrnaseq`
- Bulk genomics (e.g., bulk RNA-seq, WGS/WXS): `--database-modality bulk_genomics`
- Spatial transcriptomics: `--database-modality spatial_transcriptomics`

To enable parallel extraction, increase `--num-threads` (for example `--num-threads 8`).


## To add more modalities (e.g., genomics, pathology):
1. Define new pubmed query and extraction instructions in `radiology_dataset_db/config.py`
2. Implement new dataset schema class and extraction function in `radiology_dataset_db/extract_MODALITY_dataset_information_llm.py`
3. Import and call the new extraction function in `scripts/build_db.py` and add a conditional to check the modality type
4. Optionally, update .github/workflows/update_dbs.yml to run the pipeline for the new modality on a schedule
5. Optionally, add some ground truth papers to tests/conftest.py, and add to get_modality_info in tests/test_llm_output.py to check that the new extraction function is working as expected
All instructions are notaded in the code with comments like `#* add additional extraction instructions and functions for other modalities here, e.g. genomics, pathology, etc`

Example codex prompt used to add a scRNA-seq dataset schema and extraction function:
```text
Pleas write a module very similar to extract_radiology_dataset_information_llm.py called extract_scrnaseq_dataset_information_llm.py that looks for scRNA-seq/snRNA-seq data. It should look for name, num_patients, sequencing_technology (eg 10X, SMARTSEQ, Parse, etc), disease, species, tissue, cell/nuclei. Also have fields for paper_title, paper_link, paper_year etc that get populated afterwards. Add instructions in config.py, and add an extra condition to build_db.py (areas to edit are marked by "#*"). Add integration test structure in test_llm_output.py and add a placeholderground truth paper to conftest.py to test the new extraction function.
```

## Testing
### Just unit tests:
`pytest`

### Just integration tests:
`pytest -m integration`
(not all tests need to pass because LLM has some randomness, but most should pass consistently)

### All tests:
`pytest -m ""`
