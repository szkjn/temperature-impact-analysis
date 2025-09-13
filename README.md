# Temperature vs Singularity

A project analyzing temperature vs singularity using French language processing with spaCy, FastText, and lexical richness metrics.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environment handling.

### Prerequisites

Make sure you have uv installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies and create virtual environment:
   ```bash
   uv sync
   ```

3. Install the French spaCy language model:
   ```bash
   uv run python -m spacy download fr_core_news_lg
   ```

### Usage

#### Text Analysis Module

The `text_analysis.py` module provides functions for French text analysis:

```python
from text_analysis import clean_text, get_language_confidence, get_vocabulary_coverage, calculate_mtld

# Clean text
cleaned = clean_text("Bonjour * le monde")

# Language detection
lang, confidence = get_language_confidence("Bonjour le monde")

# French vocabulary coverage
coverage = get_vocabulary_coverage("Bonjour le monde")

# Lexical diversity (MTLD)
diversity = calculate_mtld("Bonjour le monde")
```

#### Jupyter Notebook Analysis

Run the analysis notebook:
```bash
uv run jupyter notebook metrics.ipynb
```

The notebook uses the text analysis functions to analyze text quality metrics including:
- Language detection confidence
- French vocabulary coverage
- Lexical diversity (MTLD)
- Text preprocessing

### Development

Install development dependencies:
```bash
uv sync --dev
```

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black .
```

Lint code:
```bash
uv run flake8
```
