# Vies de Genero Paper

This repository consolidates the manuscript, analysis code, notebook workflow, datasets, and supporting materials for the paper "Reversal and Persistence of Gender Biases in GPT Models" / "Reversao e Persistencia de Vieses de Genero em Modelos GPT".

## Structure

- `paper/latex/`: LaTeX sources, bibliography, and figure assets.
- `paper/pdf/`: Compiled PDF outputs and figure bundles.
- `analysis/scripts/`: Reproducible analysis scripts.
- `analysis/notebooks/`: Notebook-based experimentation and generation workflow.
- `data/raw/`: Unified raw datasets used by the analysis pipeline.
- `data/derived/`: Main regression and proportion outputs.
- `data/supporting/`: Extra derived tables and consolidated regression summaries.
- `docs/`: Supporting submission documents.
- `presentations/`: Presentation decks related to the paper.
- `archive/`: Original source bundle preserved for reference.

## Primary Files

- Portuguese manuscript: `paper/latex/main_portuguese.tex`
- English manuscript: `paper/latex/main_english.tex`
- Analysis script: `analysis/scripts/tcc_complete_analysis.py`
- Main notebook: `analysis/notebooks/march_2026_tcc_publicavel.ipynb`

## Provenance

The repository was assembled from:

- `C:\prova-ai\Working paper data`
- `C:\Users\otavi\Downloads\Copia_de_March_2026_TCC_E_PUBLICAVEL_.ipynb`
- Additional supporting files in `C:\Users\otavi\Downloads`

Filenames were normalized where useful for repository readability, but content was copied without substantive edits.

## Notes

- The LaTeX figure path expects `figuras_final/` to remain adjacent to the main `.tex` files.
- The notebook still contains references to Google Colab and local Drive paths from the original workflow.
- The analysis script uses container-style input and output paths and may need small path adjustments before local reruns.