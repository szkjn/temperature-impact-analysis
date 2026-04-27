import os
import urllib.request

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lexicalrichness import LexicalRichness
from sklearn.metrics.pairwise import cosine_similarity

_ft_model = None


def _get_ft_model():
    global _ft_model
    if _ft_model is None:
        model_path = "lid.176.bin"
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                model_path,
            )
        _ft_model = fasttext.load_model(model_path)
    return _ft_model


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
    except Exception:
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


def calculate_mtld(text):
    """
    Calculate MTLD (Measure of Textual Lexical Diversity) using scientific method
    """
    if not text.strip():
        return 0.0
    
    try:
        lex = LexicalRichness(text)
        return lex.mtld(threshold=0.72)  # Standard threshold
    except Exception:
        return 0.0


def calculate_robust_stats(values):
    """
    Calculate robust statistics for a series of values.
    
    Args:
        values: Array-like of numerical values
        
    Returns:
        dict: Dictionary with robust statistics
            - median: Median value
            - mad: Median Absolute Deviation
            - q1: First quartile (25th percentile)
            - q3: Third quartile (75th percentile)
            - iqr: Interquartile range (Q3 - Q1)
            - iqr_lower: Q1 - 1.5*IQR (lower fence)
            - iqr_upper: Q3 + 1.5*IQR (upper fence)
    """
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN values
    
    if len(values) == 0:
        return {
            'median': 0.0, 'mad': 0.0, 'q1': 0.0, 'q3': 0.0, 
            'iqr': 0.0, 'iqr_lower': 0.0, 'iqr_upper': 0.0
        }
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    
    return {
        'median': median,
        'mad': mad,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'iqr_lower': iqr_lower,
        'iqr_upper': iqr_upper
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
    _, lang_confidence = get_language_confidence(text)
    
    
    # STEP 4: METRIC MTLD
    mtld_score = calculate_mtld(text)
    
    # STEP 5: METRIC Semantic similarity
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
        'mtld': mtld_score,
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
                'mtld': metrics['mtld'],
                'semantic_similarity': metrics['semantic_similarity']
            }
            results.append(ordered_metrics)
    
    analysis_df = pd.DataFrame(results)
    
    # Sort by temperature (low to high)
    analysis_df = analysis_df.sort_values('temperature').reset_index(drop=True)
    
    print(f"Analyzed {len(analysis_df)} samples for {model_name}")
    print(f"  Temperature distribution: {sorted(analysis_df['temperature'].unique())}")
    
    return analysis_df


def calculate_temperature_stats(analysis_df, include_robust=True):
    """
    Calculate aggregated statistics by temperature.
    
    Args:
        analysis_df (pd.DataFrame): DataFrame from process_model_data()
        include_robust (bool): If True, include robust statistics (median, MAD, IQR)
        
    Returns:
        pd.DataFrame: Aggregated statistics with flattened column names
    """
    # Standard statistics (mean, std)
    temp_stats = analysis_df.groupby('temperature').agg({
        'language_confidence': ['mean', 'std'],
        'mtld': ['mean', 'std'],
        'semantic_similarity': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names for standard stats
    temp_stats.columns = ['temperature', 'lang_conf_mean', 'lang_conf_std', 
                         'mtld_mean', 'mtld_std', 'semantic_mean', 'semantic_std']
    
    # Add robust statistics if requested
    if include_robust:
        metrics = ['language_confidence', 'mtld', 'semantic_similarity']
        metric_names = ['lang_conf', 'mtld', 'semantic']
        
        for temp in temp_stats['temperature'].unique():
            temp_data = analysis_df[analysis_df['temperature'] == temp]
            temp_idx = temp_stats[temp_stats['temperature'] == temp].index[0]
            
            for metric, metric_name in zip(metrics, metric_names):
                robust_stats = calculate_robust_stats(temp_data[metric])
                
                # Add robust statistics to the dataframe
                temp_stats.loc[temp_idx, f'{metric_name}_median'] = robust_stats['median']
                temp_stats.loc[temp_idx, f'{metric_name}_mad'] = robust_stats['mad']
                temp_stats.loc[temp_idx, f'{metric_name}_q1'] = robust_stats['q1']
                temp_stats.loc[temp_idx, f'{metric_name}_q3'] = robust_stats['q3']
                temp_stats.loc[temp_idx, f'{metric_name}_iqr'] = robust_stats['iqr']
    
    return temp_stats


PLOT_LABELS = {
    'fr': {
        'median': 'Médiane',
        'mean': 'Moyenne',
        'suptitle': 'Température vs Métriques',
        'lang_title': 'Intégrité linguistique (fastText)',
        'lang_xlabel': 'Température',
        'lang_ylabel': 'Confiance',
        'mtld_title': 'Diversité lexicale (MTLD)',
        'mtld_xlabel': 'Température',
        'mtld_ylabel': 'MTLD',
        'sem_title': 'Similarité sémantique (Sentence-BERT)',
        'sem_xlabel': 'Température',
        'sem_ylabel': 'Similarité',
        'sweet_zone': 'Zone de singularisation',
        'break_zone': 'Régime de rupture',
        'legend_title': 'Zones de température',
    },
    'en': {
        'median': 'Median',
        'mean': 'Mean',
        'suptitle': 'Temperature vs Metrics',
        'lang_title': 'Linguistic integrity (fastText)',
        'lang_xlabel': 'Temperature',
        'lang_ylabel': 'Confidence',
        'mtld_title': 'Lexical diversity (MTLD)',
        'mtld_xlabel': 'Temperature',
        'mtld_ylabel': 'MTLD',
        'sem_title': 'Semantic similarity (Sentence-BERT)',
        'sem_xlabel': 'Temperature',
        'sem_ylabel': 'Similarity',
        'sweet_zone': 'Singularity zone',
        'break_zone': 'Breakdown regime',
        'legend_title': 'Temperature zones',
    },
}


def plot_single_model_metrics(temp_stats, model_name, sample_size=30, error_type='ci', robust_center='mean',
                              show_zones=False, sweet_spot=(1, 1.4), degradation_threshold=1.5,
                              figure_caption=None, language='fr'):
    """
    Create visualization for a single model's metrics with multiple error bar options.
    
    Args:
        temp_stats (pd.DataFrame): Temperature statistics
        model_name (str): Name of the model for the title
        sample_size (int): Number of samples per temperature point (default: 30)
        error_type (str): Type of error bars to use:
            - 'ci': 95% confidence intervals (default)
            - 'std': Standard deviation
            - 'mad': Median Absolute Deviation (robust)
            - 'iqr': Interquartile range (robust)
        robust_center (str): Central tendency measure:
            - 'mean': Use mean (default)
            - 'median': Use median (robust)
        show_zones (bool): If True, highlight sweet spot and degradation zones
        sweet_spot (tuple): Temperature range for optimal zone (default: 1.2-1.4)
        degradation_threshold (float): Temperature where degradation begins (default: 1.4)
    
    Returns:
        tuple: (fig, axes) matplotlib objects
    """
    # Determine central values
    if robust_center == 'median':
        # Check if robust statistics are available
        if 'lang_conf_median' not in temp_stats.columns:
            print("Warning: Robust statistics not available. Using mean instead.")
            robust_center = 'mean'
    
    labels = PLOT_LABELS[language]

    if robust_center == 'median':
        lang_center = temp_stats['lang_conf_median']
        mtld_center = temp_stats['mtld_median']
        semantic_center = temp_stats['semantic_median']
        center_label = labels['median']
    else:
        lang_center = temp_stats['lang_conf_mean']
        mtld_center = temp_stats['mtld_mean']
        semantic_center = temp_stats['semantic_mean']
        center_label = labels['mean']
    
    # Calculate error bars based on error_type
    if error_type == 'ci':
        ci_multiplier = 1.96  # 95% confidence interval
        lang_err = ci_multiplier * temp_stats['lang_conf_std'] / np.sqrt(sample_size)
        mtld_err = ci_multiplier * temp_stats['mtld_std'] / np.sqrt(sample_size)
        semantic_err = ci_multiplier * temp_stats['semantic_std'] / np.sqrt(sample_size)
        error_label = "95% Confidence Intervals"
    elif error_type == 'std':
        lang_err = temp_stats['lang_conf_std']
        mtld_err = temp_stats['mtld_std']
        semantic_err = temp_stats['semantic_std']
        error_label = "Standard Deviation"
    elif error_type == 'mad':
        if 'lang_conf_mad' not in temp_stats.columns:
            print("Warning: MAD statistics not available. Using standard deviation instead.")
            error_type = 'std'
            lang_err = temp_stats['lang_conf_std']
            mtld_err = temp_stats['mtld_std']
            semantic_err = temp_stats['semantic_std']
            error_label = "Standard Deviation (MAD not available)"
        else:
            lang_err = temp_stats['lang_conf_mad']
            mtld_err = temp_stats['mtld_mad']
            semantic_err = temp_stats['semantic_mad']
            error_label = "Median Absolute Deviation (MAD)"
    elif error_type == 'iqr':
        if 'lang_conf_iqr' not in temp_stats.columns:
            print("Warning: IQR statistics not available. Using standard deviation instead.")
            error_type = 'std'
            lang_err = temp_stats['lang_conf_std']
            mtld_err = temp_stats['mtld_std']
            semantic_err = temp_stats['semantic_std']
            error_label = "Standard Deviation (IQR not available)"
        else:
            # For IQR, we use half the IQR as error bars (Q1 to Q3 range)
            lang_err = temp_stats['lang_conf_iqr'] / 2
            mtld_err = temp_stats['mtld_iqr'] / 2
            semantic_err = temp_stats['semantic_iqr'] / 2
            error_label = "Interquartile Range (IQR)"
    else:
        raise ValueError(f"Unknown error_type: {error_type}. Use 'ci', 'std', 'mad', or 'iqr'.")
    
    # Create plots - 2x2 layout with 3 plots (2 on top, 1 on bottom)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{labels['suptitle']}: {model_name} ({center_label} ± {error_label})")
    
    # Flatten axes for easier indexing, but we'll only use first 3
    axes_flat = axes.flatten()
    
    # Add zones if requested
    if show_zones:
        temp_min = temp_stats['temperature'].min()
        temp_max = temp_stats['temperature'].max()
        
        # Apply zones to first 3 plots only
        for i in range(3):
            ax = axes_flat[i]
            # Singularity / sweet-spot zone — green
            if sweet_spot[0] <= temp_max and sweet_spot[1] >= temp_min:
                ax.axvspan(max(sweet_spot[0], temp_min), min(sweet_spot[1], temp_max),
                          alpha=0.35, color='green', zorder=0,
                          label=f"{labels['sweet_zone']} ({sweet_spot[0]}-{sweet_spot[1]})")

            # Breakdown regime — red
            if degradation_threshold <= temp_max:
                ax.axvspan(max(degradation_threshold, temp_min), temp_max,
                          alpha=0.35, color='red', zorder=0,
                          label=f"{labels['break_zone']} (≥{degradation_threshold})")
    
    # Language confidence (top left)
    axes_flat[0].fill_between(temp_stats['temperature'], 
                         lang_center - lang_err, 
                         lang_center + lang_err, 
                         alpha=0.1)
    axes_flat[0].plot(temp_stats['temperature'], lang_center, 'o-', linewidth=1.5, markersize=3)
    axes_flat[0].set_title(labels['lang_title'])
    axes_flat[0].set_xlabel(labels['lang_xlabel'])
    axes_flat[0].set_ylabel(labels['lang_ylabel'])
    axes_flat[0].grid(True, alpha=0.3)
    
    # MTLD (top right)
    axes_flat[1].fill_between(temp_stats['temperature'], 
                         mtld_center - mtld_err, 
                         mtld_center + mtld_err, 
                         alpha=0.1)
    axes_flat[1].plot(temp_stats['temperature'], mtld_center, 'o-', linewidth=1.5, markersize=3)
    axes_flat[1].set_title(labels['mtld_title'])
    axes_flat[1].set_xlabel(labels['mtld_xlabel'])
    axes_flat[1].set_ylabel(labels['mtld_ylabel'])
    axes_flat[1].grid(True, alpha=0.3)
    
    # Semantic similarity (bottom left)
    axes_flat[2].fill_between(temp_stats['temperature'], 
                         semantic_center - semantic_err, 
                         semantic_center + semantic_err, 
                         alpha=0.1)
    axes_flat[2].plot(temp_stats['temperature'], semantic_center, 'o-', linewidth=1.5, markersize=3)
    axes_flat[2].set_title(labels['sem_title'])
    axes_flat[2].set_xlabel(labels['sem_xlabel'])
    axes_flat[2].set_ylabel(labels['sem_ylabel'])
    axes_flat[2].grid(True, alpha=0.3)
    
    # Use the bottom right subplot for the common legend
    axes_flat[3].axis('off')  # Turn off axis
    
    if show_zones:
        # Create legend handles manually
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.35, label=f"{labels['sweet_zone']} ({sweet_spot[0]}-{sweet_spot[1]})"),
            Patch(facecolor='red', alpha=0.35, label=f"{labels['break_zone']} (≥{degradation_threshold})")
        ]

        # Place legend in the bottom right subplot
        axes_flat[3].legend(handles=legend_elements, loc='center', fontsize=14,
                           title=labels['legend_title'], title_fontsize=16,
                           frameon=False, handlelength=4, handleheight=2.5,
                           labelspacing=1.2)
    
    plt.tight_layout()
    return fig, axes


def plot_temperature_zone(temp_stats, model_name, temp_min=0.75, temp_max=1.5, sample_size=30, error_type='ci', robust_center='mean', 
                          show_zones=False, sweet_spot=(1, 1.4), degradation_threshold=1.5):
    """
    Create visualization for a specific temperature range (zone of interest).
    
    Args:
        temp_stats (pd.DataFrame): Temperature statistics
        model_name (str): Name of the model for the title
        temp_min (float): Minimum temperature to display (default: 0.75)
        temp_max (float): Maximum temperature to display (default: 1.5)
        sample_size (int): Number of samples per temperature point (default: 30)
        error_type (str): Type of error bars to use:
            - 'ci': 95% confidence intervals (default)
            - 'std': Standard deviation
            - 'mad': Median Absolute Deviation (robust)
            - 'iqr': Interquartile range (robust)
        robust_center (str): Central tendency measure:
            - 'mean': Use mean (default)
            - 'median': Use median (robust)
    
    Returns:
        tuple: (fig, axes) matplotlib objects
    """
    # Filter data to temperature range
    filtered_stats = temp_stats[
        (temp_stats['temperature'] >= temp_min) & 
        (temp_stats['temperature'] <= temp_max)
    ].copy()
    
    if len(filtered_stats) == 0:
        print(f"No data found in temperature range {temp_min}-{temp_max}")
        return None, None
    
    # Use the same logic as plot_single_model_metrics but with filtered data
    fig, axes = plot_single_model_metrics(
        filtered_stats, 
        f"{model_name} (Zone {temp_min}-{temp_max})", 
        sample_size=sample_size, 
        error_type=error_type, 
        robust_center=robust_center,
        show_zones=show_zones,
        sweet_spot=sweet_spot,
        degradation_threshold=degradation_threshold
    )
    
    # Adjust x-axis limits for the zone
    for ax in axes:
        ax.set_xlim(temp_min - 0.05, temp_max + 0.05)
    
    # Print some stats about the filtered data
    print(f"Temperature range: {temp_min} - {temp_max}")
    print(f"Data points: {len(filtered_stats)}")
    print(f"Temperature values: {sorted(filtered_stats['temperature'].tolist())}")
    
    return fig, axes


def export_temperature_stats_to_csv(temp_stats, model_name, filename=None, temp_min=None, temp_max=None):
    """
    Export temperature statistics to a CSV file.
    
    Args:
        temp_stats (pd.DataFrame): Temperature statistics from calculate_temperature_stats()
        model_name (str): Name of the model for the filename
        filename (str, optional): Custom filename. If None, auto-generates based on model name
        temp_min (float, optional): Minimum temperature to filter (inclusive)
        temp_max (float, optional): Maximum temperature to filter (inclusive)
    
    Returns:
        str: Path to the exported CSV file
    """
    # Filter data if temperature range is specified
    data_to_export = temp_stats.copy()
    if temp_min is not None or temp_max is not None:
        if temp_min is not None:
            data_to_export = data_to_export[data_to_export['temperature'] >= temp_min]
        if temp_max is not None:
            data_to_export = data_to_export[data_to_export['temperature'] <= temp_max]
        
        # Update filename to reflect the range
        range_suffix = f"_temp_{temp_min or 'min'}-{temp_max or 'max'}"
    else:
        range_suffix = ""
    
    # Generate filename if not provided
    if filename is None:
        # Clean model name for filename
        clean_model_name = model_name.lower().replace(' ', '_').replace('.', '_').replace('-', '_')
        filename = f"temperature_stats_{clean_model_name}{range_suffix}.csv"
    
    # Export to CSV
    data_to_export.to_csv(filename, index=False, float_format='%.6f')
    
    # Print summary
    print(f"Exported temperature statistics to: {filename}")
    print(f"Model: {model_name}")
    print(f"Data points: {len(data_to_export)}")
    if temp_min is not None or temp_max is not None:
        print(f"Temperature range: {temp_min or 'min'} - {temp_max or 'max'}")
        print(f"Temperature values: {sorted(data_to_export['temperature'].tolist())}")
    else:
        temp_range = f"{data_to_export['temperature'].min():.1f} - {data_to_export['temperature'].max():.1f}"
        print(f"Full temperature range: {temp_range}")
    
    return filename

