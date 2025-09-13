import os
import spacy
import fasttext
from lexicalrichness import LexicalRichness
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from collections import Counter

# Load models (lazy loading)
_ft_model = None
_nlp = None

def _get_ft_model():
    global _ft_model
    if _ft_model is None:
        model_path = "lid.176.bin"
        if not os.path.exists(model_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", 
                model_path
            )
        _ft_model = fasttext.load_model(model_path)
    return _ft_model

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("fr_core_news_lg")
    return _nlp

def get_language_confidence(text):
    """
    Get language identification confidence using FastText (Joulin et al., 2017).
    
    Returns:
        tuple: (detected_language, confidence_score)
        
    Reference:
        Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). 
        Bag of tricks for efficient text classification. EACL 2017.
    """
    try:
        ft_model = _get_ft_model()
        labels, scores = ft_model.predict(text.replace('\n', ' '), k=1)
        detected_lang = labels[0].replace('__label__', '')
        confidence = scores[0]
        return detected_lang, confidence
    except:
        return "unknown", 0.0

def clean_text(text):
    """
    Clean text by removing asterisks and normalizing whitespace.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove asterisks
    cleaned = text.replace('*', '')
    
    # Replace newlines with spaces
    cleaned = cleaned.replace('\n', ' ')
    
    # Normalize multiple whitespaces to single spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def get_vocabulary_coverage(text):
    """
    Calculate the proportion of tokens recognized by the French spaCy model.
    This measures how much of the text consists of valid French vocabulary.
    
    Returns:
        float: Proportion of recognized French tokens (0.0 to 1.0)
        
    Note: 
        Uses spaCy's fr_core_news_lg model which is trained on French corpora.
        OOV (out-of-vocabulary) tokens indicate non-standard or corrupted words.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    alpha_tokens = [token for token in doc if token.is_alpha]
    if not alpha_tokens:
        return 0.0
    
    recognized_tokens = [token for token in alpha_tokens if not token.is_oov]
    return len(recognized_tokens) / len(alpha_tokens)

def calculate_mtld(text):
    """
    Calculate MTLD (Measure of Textual Lexical Diversity) using scientific method
    """
    if not text.strip():
        return 0.0
    
    try:
        lex = LexicalRichness(text)
        return lex.mtld(threshold=0.72)  # Standard threshold
    except:
        return 0.0


def _shannon_entropy(counts):
    """Return Shannon entropy H in bits from a Counter of symbol counts."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log2(p)
    return H


def get_output_entropy(text, level="token", normalize=True):
    """
    Measure output entropy as a proxy for randomness/noise.

    Args:
        text (str): the generated text
        level (str): "char", "token", or "both"
        normalize (bool): if True, divides H by log2(V) where V is the
                          number of unique symbols/tokens, yielding [0,1].

    Returns:
        float or dict: normalized entropy (if normalize=True) or raw bits.
                       If level="both", returns:
                       {
                         "char": {"raw": Hc, "norm": Hc_norm},
                         "token": {"raw": Ht, "norm": Ht_norm}
                       }
    """
    s = clean_text(text)

    # --- Character-level (exclude whitespace to avoid length artefacts) ---
    chars = [ch for ch in s if not ch.isspace()]
    char_counts = Counter(chars)
    Hc = _shannon_entropy(char_counts)
    Hc_norm = Hc / math.log2(len(char_counts)) if len(char_counts) > 1 else 0.0

    # --- Token-level (alpha tokens) ---
    tokens = None
    try:
        # Prefer spaCy tokenization if model already loaded
        nlp = _get_nlp()
        doc = nlp(s)
        tokens = [t.text.lower() for t in doc if t.is_alpha]
    except Exception:
        # Fallback: simple regex word tokenizer
        tokens = re.findall(r"\p{L}+(?:'\p{L}+)?", s, flags=re.UNICODE)

    tok_counts = Counter(tokens)
    Ht = _shannon_entropy(tok_counts)
    Ht_norm = Ht / math.log2(len(tok_counts)) if len(tok_counts) > 1 else 0.0

    if level == "char":
        return Hc_norm if normalize else Hc
    elif level == "token":
        return Ht_norm if normalize else Ht
    else:  # "both"
        return {
            "char": {"raw": Hc, "norm": Hc_norm},
            "token": {"raw": Ht, "norm": Ht_norm},
        }


def simple_analysis(row, sbert_model):
    """
    Perform comprehensive text analysis on a single row of data.
    
    Args:
        row: DataFrame row or dict containing 'text' and optionally 'user_prompt'
        sbert_model: Pre-loaded SentenceTransformer model for semantic similarity
        
    Returns:
        dict: Analysis metrics or None if analysis fails
    """
    text = row if isinstance(row, str) else row.get('text', '')
    
    # STEP 1: CLEAN TEXT
    text = clean_text(text)
    
    # STEP 2: METRIC Language confidence
    detected_lang, lang_confidence = get_language_confidence(text)
    
    # STEP 3: METRIC Vocabulary coverage  
    vocab_coverage = get_vocabulary_coverage(text)
    
    # STEP 4: METRIC MTLD
    mtld_score = calculate_mtld(text)
    
    # STEP 5: METRIC Output entropy (character and token level)
    char_entropy = get_output_entropy(text, level="char", normalize=True)
    token_entropy = get_output_entropy(text, level="token", normalize=True)
    
    # STEP 6: METRIC Semantic similarity
    semantic_similarity = 0.0
    if not isinstance(row, str):
        prompt = row.get('user_prompt', '')
        if prompt.strip() and text.strip():
            try:
                embeddings = sbert_model.encode([prompt, text])
                semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                semantic_similarity = max(0.0, float(semantic_similarity))
            except Exception as e:
                print(f"Error calculating semantic similarity: {e}")
                semantic_similarity = 0.0
    
    return {
        'language_confidence': lang_confidence,
        'vocab_coverage': vocab_coverage, 
        'mtld': mtld_score,
        'char_entropy': char_entropy,
        'token_entropy': token_entropy,
        'semantic_similarity': semantic_similarity
    }


def process_model_data(df, sbert_model, model_name="Model"):
    """
    Process a DataFrame of model outputs and calculate metrics for each sample.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['temperature', 'iteration', 'text', 'user_prompt']
        sbert_model: Pre-loaded SentenceTransformer model for semantic similarity
        model_name (str): Name of the model for logging purposes
        
    Returns:
        pd.DataFrame: Processed DataFrame with analysis metrics
    """
    results = []
    
    print(f"Processing {len(df)} samples for {model_name}...")
    
    for i, row in df.iterrows():
        if i % 100 == 0:  # Progress indicator
            print(f"  Processed {i}/{len(df)} samples...")
            
        metrics = simple_analysis(row, sbert_model)
        if metrics:
            ordered_metrics = {
                'temperature': row['temperature'],
                'iteration': row['iteration'],
                'language_confidence': metrics['language_confidence'],
                'vocab_coverage': metrics['vocab_coverage'],
                'mtld': metrics['mtld'],
                'char_entropy': metrics['char_entropy'],
                'token_entropy': metrics['token_entropy'],
                'semantic_similarity': metrics['semantic_similarity']
            }
            results.append(ordered_metrics)
    
    analysis_df = pd.DataFrame(results)
    
    # Sort by temperature (low to high)
    analysis_df = analysis_df.sort_values('temperature').reset_index(drop=True)
    
    print(f"✓ Analyzed {len(analysis_df)} samples for {model_name}")
    print(f"  Temperature distribution: {sorted(analysis_df['temperature'].unique())}")
    
    return analysis_df


def calculate_temperature_stats(analysis_df):
    """
    Calculate aggregated statistics by temperature.
    
    Args:
        analysis_df (pd.DataFrame): DataFrame from process_model_data()
        
    Returns:
        pd.DataFrame: Aggregated statistics with flattened column names
    """
    temp_stats = analysis_df.groupby('temperature').agg({
        'language_confidence': ['mean', 'std'],
        'vocab_coverage': ['mean', 'std'],
        'mtld': ['mean', 'std'],
        'char_entropy': ['mean', 'std'],
        'token_entropy': ['mean', 'std'],
        'semantic_similarity': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    temp_stats.columns = ['temperature', 'lang_conf_mean', 'lang_conf_std', 
                         'vocab_mean', 'vocab_std', 'mtld_mean', 'mtld_std',
                         'char_entropy_mean', 'char_entropy_std', 
                         'token_entropy_mean', 'token_entropy_std',
                         'semantic_mean', 'semantic_std']
    
    return temp_stats


def plot_single_model_metrics(temp_stats, model_name, sample_size=30, use_ci=True, include_entropy=True):
    """
    Create visualization for a single model's metrics.
    
    Args:
        temp_stats (pd.DataFrame): Temperature statistics with columns including entropy metrics
        model_name (str): Name of the model for the title
        sample_size (int): Number of samples per temperature point (default: 30)
        use_ci (bool): If True, use 95% confidence intervals; if False, use standard deviation
        include_entropy (bool): If True, show 6 plots including entropy; if False, show 4 original plots
    
    Returns:
        tuple: (fig, axes) matplotlib objects
    """    
    # Calculate error bars
    if use_ci:
        ci_multiplier = 1.96  # 95% confidence interval
        lang_err = ci_multiplier * temp_stats['lang_conf_std'] / np.sqrt(sample_size)
        vocab_err = ci_multiplier * temp_stats['vocab_std'] / np.sqrt(sample_size)
        mtld_err = ci_multiplier * temp_stats['mtld_std'] / np.sqrt(sample_size)
        semantic_err = ci_multiplier * temp_stats['semantic_std'] / np.sqrt(sample_size)
        error_type = "95% Confidence Intervals"
        if include_entropy:
            char_entropy_err = ci_multiplier * temp_stats['char_entropy_std'] / np.sqrt(sample_size)
            token_entropy_err = ci_multiplier * temp_stats['token_entropy_std'] / np.sqrt(sample_size)
    else:
        lang_err = temp_stats['lang_conf_std']
        vocab_err = temp_stats['vocab_std']
        mtld_err = temp_stats['mtld_std']
        semantic_err = temp_stats['semantic_std']
        error_type = "Standard Deviation"
        if include_entropy:
            char_entropy_err = temp_stats['char_entropy_std']
            token_entropy_err = temp_stats['token_entropy_std']
    
    # Create plots
    n_plots = 6 if include_entropy else 4
    fig_width = 30 if include_entropy else 20
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 5))
    fig.suptitle(f'Temperature vs Metrics: {model_name} ({error_type})')
    
    # Language confidence
    axes[0].errorbar(temp_stats['temperature'], temp_stats['lang_conf_mean'], 
                     yerr=lang_err, marker='o', capsize=5)
    axes[0].set_title('FastText Language Confidence\n(Joulin et al., 2017)')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Confidence')
    axes[0].grid(True, alpha=0.3)
    
    # Vocabulary coverage
    axes[1].errorbar(temp_stats['temperature'], temp_stats['vocab_mean'], 
                     yerr=vocab_err, marker='o', capsize=5)
    axes[1].set_title('French Vocabulary Coverage\n(OOV Analysis)')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Coverage')
    axes[1].grid(True, alpha=0.3)
    
    # MTLD
    axes[2].errorbar(temp_stats['temperature'], temp_stats['mtld_mean'], 
                     yerr=mtld_err, marker='o', capsize=5)
    axes[2].set_title('MTLD Lexical Diversity\n(McCarthy & Jarvis, 2010)')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('MTLD')
    axes[2].grid(True, alpha=0.3)
    
    if include_entropy:
        # Character entropy
        axes[3].errorbar(temp_stats['temperature'], temp_stats['char_entropy_mean'], 
                         yerr=char_entropy_err, marker='o', capsize=5)
        axes[3].set_title('Character Entropy\n(Shannon, normalized)')
        axes[3].set_xlabel('Temperature')
        axes[3].set_ylabel('Normalized Entropy')
        axes[3].grid(True, alpha=0.3)
        
        # Token entropy
        axes[4].errorbar(temp_stats['temperature'], temp_stats['token_entropy_mean'], 
                         yerr=token_entropy_err, marker='o', capsize=5)
        axes[4].set_title('Token Entropy\n(Shannon, normalized)')
        axes[4].set_xlabel('Temperature')
        axes[4].set_ylabel('Normalized Entropy')
        axes[4].grid(True, alpha=0.3)
        
        # Semantic similarity
        axes[5].errorbar(temp_stats['temperature'], temp_stats['semantic_mean'], 
                         yerr=semantic_err, marker='o', capsize=5)
        axes[5].set_title('Semantic Similarity\n(Reimers & Gurevych, 2019)')
        axes[5].set_xlabel('Temperature')
        axes[5].set_ylabel('Similarity')
        axes[5].grid(True, alpha=0.3)
    else:
        # Semantic similarity (position 3 when no entropy)
        axes[3].errorbar(temp_stats['temperature'], temp_stats['semantic_mean'], 
                         yerr=semantic_err, marker='o', capsize=5)
        axes[3].set_title('Semantic Similarity\n(Reimers & Gurevych, 2019)')
        axes[3].set_xlabel('Temperature')
        axes[3].set_ylabel('Similarity')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_combined_models_metrics(models_data, temp_range=None, use_ci=True, include_entropy=True):
    """
    Create combined visualization for multiple models' metrics.
    
    Args:
        models_data (dict): Dictionary with model data in format:
            {
                'model_name': {
                    'temp_stats': pd.DataFrame,
                    'sample_size': int,
                    'color': str,
                    'marker': str
                }
            }
        temp_range (tuple): Optional (min_temp, max_temp) to filter data
        use_ci (bool): If True, use 95% confidence intervals; if False, use standard deviation
        include_entropy (bool): If True, show 6 plots including entropy; if False, show 4 original plots
    
    Returns:
        tuple: (fig, axes) matplotlib objects
        
    Example:
        models_data = {
            'OpenAI': {
                'temp_stats': temp_stats_openai,
                'sample_size': 30,
                'color': 'blue',
                'marker': 'o'
            },
            'Mistral': {
                'temp_stats': temp_stats_mistral,
                'sample_size': 30,
                'color': 'red', 
                'marker': 's'
            },
            'Anthropic': {
                'temp_stats': temp_stats_anthropic,
                'sample_size': 30,
                'color': 'green',
                'marker': '^'
            }
        }
    """
    error_type = "95% Confidence Intervals" if use_ci else "Standard Deviation"
    temp_suffix = f" ({temp_range[0]:.1f}-{temp_range[1]:.1f})" if temp_range else ""
    
    # Create plots
    n_plots = 6 if include_entropy else 4
    fig_width = 30 if include_entropy else 20
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 5))
    fig.suptitle(f'Temperature vs Metrics: Model Comparison{temp_suffix} ({error_type})')
    
    for model_name, model_info in models_data.items():
        temp_stats = model_info['temp_stats'].copy()
        sample_size = model_info['sample_size']
        color = model_info['color']
        marker = model_info['marker']
        
        # Filter by temperature range if specified
        if temp_range:
            temp_stats = temp_stats[
                (temp_stats['temperature'] >= temp_range[0]) & 
                (temp_stats['temperature'] <= temp_range[1])
            ]
        
        # Calculate error bars
        if use_ci:
            ci_multiplier = 1.96  # 95% confidence interval
            lang_err = ci_multiplier * temp_stats['lang_conf_std'] / np.sqrt(sample_size)
            vocab_err = ci_multiplier * temp_stats['vocab_std'] / np.sqrt(sample_size)
            mtld_err = ci_multiplier * temp_stats['mtld_std'] / np.sqrt(sample_size)
            semantic_err = ci_multiplier * temp_stats['semantic_std'] / np.sqrt(sample_size)
            if include_entropy:
                char_entropy_err = ci_multiplier * temp_stats['char_entropy_std'] / np.sqrt(sample_size)
                token_entropy_err = ci_multiplier * temp_stats['token_entropy_std'] / np.sqrt(sample_size)
        else:
            lang_err = temp_stats['lang_conf_std']
            vocab_err = temp_stats['vocab_std']
            mtld_err = temp_stats['mtld_std']
            semantic_err = temp_stats['semantic_std']
            if include_entropy:
                char_entropy_err = temp_stats['char_entropy_std']
                token_entropy_err = temp_stats['token_entropy_std']
        
        # Language confidence
        axes[0].errorbar(temp_stats['temperature'], temp_stats['lang_conf_mean'], 
                         yerr=lang_err, marker=marker, capsize=5, 
                         label=model_name, color=color, alpha=0.7)
        
        # Vocabulary coverage
        axes[1].errorbar(temp_stats['temperature'], temp_stats['vocab_mean'], 
                         yerr=vocab_err, marker=marker, capsize=5, 
                         label=model_name, color=color, alpha=0.7)
        
        # MTLD
        axes[2].errorbar(temp_stats['temperature'], temp_stats['mtld_mean'], 
                         yerr=mtld_err, marker=marker, capsize=5, 
                         label=model_name, color=color, alpha=0.7)
        
        if include_entropy:
            # Character entropy
            axes[3].errorbar(temp_stats['temperature'], temp_stats['char_entropy_mean'], 
                             yerr=char_entropy_err, marker=marker, capsize=5, 
                             label=model_name, color=color, alpha=0.7)
            
            # Token entropy
            axes[4].errorbar(temp_stats['temperature'], temp_stats['token_entropy_mean'], 
                             yerr=token_entropy_err, marker=marker, capsize=5, 
                             label=model_name, color=color, alpha=0.7)
            
            # Semantic similarity
            axes[5].errorbar(temp_stats['temperature'], temp_stats['semantic_mean'], 
                             yerr=semantic_err, marker=marker, capsize=5, 
                             label=model_name, color=color, alpha=0.7)
        else:
            # Semantic similarity (position 3 when no entropy)
            axes[3].errorbar(temp_stats['temperature'], temp_stats['semantic_mean'], 
                             yerr=semantic_err, marker=marker, capsize=5, 
                             label=model_name, color=color, alpha=0.7)
    
    # Set titles and labels
    axes[0].set_title('FastText Language Confidence\n(Joulin et al., 2017)')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Confidence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('French Vocabulary Coverage\n(OOV Analysis)')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Coverage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('MTLD Lexical Diversity\n(McCarthy & Jarvis, 2010)')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('MTLD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    if include_entropy:
        axes[3].set_title('Character Entropy\n(Shannon, normalized)')
        axes[3].set_xlabel('Temperature')
        axes[3].set_ylabel('Normalized Entropy')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        axes[4].set_title('Token Entropy\n(Shannon, normalized)')
        axes[4].set_xlabel('Temperature')
        axes[4].set_ylabel('Normalized Entropy')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        axes[5].set_title('Semantic Similarity\n(Reimers & Gurevych, 2019)')
        axes[5].set_xlabel('Temperature')
        axes[5].set_ylabel('Similarity')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    else:
        axes[3].set_title('Semantic Similarity\n(Reimers & Gurevych, 2019)')
        axes[3].set_xlabel('Temperature')
        axes[3].set_ylabel('Similarity')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
