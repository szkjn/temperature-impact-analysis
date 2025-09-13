import os
import spacy
import fasttext
from lexicalrichness import LexicalRichness

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
