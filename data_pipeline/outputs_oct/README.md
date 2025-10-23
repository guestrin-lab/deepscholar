# Sampled Dataset: outputs_oct

**Created:** October 23, 2025  
**Source:** `outputs/20251015_143311/` (filtered papers)  
**Sample Size:** 100 papers  
**Sampling Method:** Stratified random sampling by primary CS category  
**Random Seed:** 42 (reproducible)

---

## 📊 Dataset Overview

This is a representative sample of 100 papers selected from 414 high-quality filtered papers using stratified random sampling to maintain the original category distribution.

### Files Included

```
outputs_oct/
├── papers.csv                # 100 sampled papers with metadata
├── paper_content.csv         # 100 papers with related works sections
├── citations/                # 100 citation CSV files (one per paper)
├── related_works/            # 100 related works CSV files
└── latex_source/             # 100 LaTeX source packages
    ├── *.tar.gz              # Compressed source archives
    └── */                    # Extracted LaTeX projects
```

### Statistics

- **Total Papers:** 100
- **Citation Files:** 100 (100%)
- **Related Works Files:** 100 (100%)
- **LaTeX Sources:** 100 (100%)

---

## 🎯 Category Distribution

Papers were sampled proportionally from each category to maintain the original distribution:

| Category | Papers | % | Description |
|----------|--------|---|-------------|
| cs.CV | 29 | 29.0% | Computer Vision |
| cs.LG | 17 | 17.0% | Machine Learning |
| cs.SE | 9 | 9.0% | Software Engineering |
| cs.HC | 8 | 8.0% | Human-Computer Interaction |
| cs.RO | 6 | 6.0% | Robotics |
| cs.DC | 5 | 5.0% | Distributed Computing |
| cs.AI | 5 | 5.0% | Artificial Intelligence |
| cs.CL | 4 | 4.0% | Computation and Language (NLP) |
| cs.NI | 4 | 4.0% | Networking |
| cs.GR | 3 | 3.0% | Graphics |
| cs.MA | 2 | 2.0% | Multiagent Systems |
| cs.CR | 2 | 2.0% | Cryptography and Security |
| cs.MM | 1 | 1.0% | Multimedia |
| cs.LO | 1 | 1.0% | Logic in CS |
| cs.PL | 1 | 1.0% | Programming Languages |
| cs.CY | 1 | 1.0% | Computers and Society |
| cs.SI | 1 | 1.0% | Social Networks |
| cs.CE | 1 | 1.0% | Computational Engineering |

---

## 📁 File Formats

### papers.csv
Contains paper metadata with columns:
- `arxiv_id`: ArXiv identifier (e.g., "2509.11056v1")
- `title`: Paper title
- `authors`: Semicolon-separated list of authors
- `abstract`: Paper abstract
- `categories`: Semicolon-separated CS categories
- `published_date`: Publication date
- `updated_date`: Last update date
- `abs_url`: ArXiv abstract URL
- `doi`: DOI (if available)
- `journal_ref`: Journal reference (if available)
- `comments`: Author comments (if available)

### paper_content.csv
Contains paper content with related works:
- `arxiv_link`: ArXiv URL
- `arxiv_id`: ArXiv identifier
- `publication_date`: Publication date
- `paper_title`: Paper title
- `abstract`: Abstract text
- `related_works_section`: Extracted related works section (clean text)
- `related_works_length`: Length in characters

### citations/[arxiv_id].csv
Individual citation files for each paper containing:
- `parent_paper_title`: The paper citing
- `parent_arxiv_link`: Parent paper URL
- `citation_shorthand`: Citation key from bibliography
- `raw_citation_text`: Raw citation from LaTeX
- `cited_paper_title`: Title of cited paper
- `cited_paper_arxiv_link`: ArXiv link (if found)
- `cited_paper_abstract`: Abstract (if found)
- `bib_paper_authors`: Authors from bibliography
- `bib_paper_year`: Year from bibliography
- `bib_paper_month`: Month from bibliography
- `bib_paper_url`: URL from bibliography
- `bib_paper_doi`: DOI from bibliography
- `bib_paper_journal`: Journal from bibliography

### related_works/[arxiv_id].csv
Related works information extracted from each paper.

### latex_source/
- **[arxiv_id].tar.gz**: Compressed LaTeX source files
- **[arxiv_id]/**: Extracted LaTeX project folders with:
  - `.tex` files
  - `.bib` files
  - Images and figures
  - Style files
  - Compiled bibliography files

---

## 🔄 Reproducibility

This dataset can be reproduced with:

```bash
python create_sample_dataset.py \
    --input-base outputs/ \
    --input-folder 20251015_143311 \
    --output-name outputs_oct \
    --sample-size 100 \
    --random-seed 42
```

---

## ✅ Quality Assurance

All papers in this dataset:
- ✅ Have a related works section
- ✅ Related works length: 200-10,000 characters
- ✅ Have at least 5 citations extracted
- ✅ Authors meet h-index criteria (min: 20)
- ✅ Complete LaTeX source available
- ✅ Citation metadata extracted

---

## 📈 Use Cases

This dataset is suitable for:
- **Training/Testing**: ML models for citation analysis
- **Benchmarking**: Related works generation systems
- **Research**: Literature review automation
- **Analysis**: Academic writing patterns
- **Evaluation**: Citation recommendation systems

---

## 🔗 Source Information

**Original Dataset:** `outputs/20251015_143311/`
- Total papers: 414 (after quality filtering)
- Original collection: 465 papers
- Quality filter pass rate: 89.03%

**Sampling Details:**
- Method: Stratified random sampling
- Strata: Primary CS category (first listed)
- Preserves: Original category proportions
- Randomization: Seeded for reproducibility

---

## 📝 Notes

- Papers can belong to multiple categories; sampling based on primary (first listed) category
- All related files (citations, LaTeX sources) are included for complete analysis
- LaTeX sources include both compressed archives and extracted folders
- Dataset is self-contained and portable

---

For questions or issues, refer to the main pipeline documentation in `data_pipeline/README.md`.

