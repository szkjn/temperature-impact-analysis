import os
import json
import spacy
import fasttext
from lexicalrichness import LexicalRichness
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load FastText language detection model (download if needed)
def load_fasttext_model():
    model_path = "lid.176.bin"
    if not os.path.exists(model_path):
        print("Downloading FastText language detection model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", 
            model_path
        )
        print("Model downloaded successfully!")
    return fasttext.load_model(model_path)


# Load models
print("Loading FastText language detection model...")
ft_model = load_fasttext_model()

print("Loading French spaCy model...")
nlp = spacy.load("fr_core_news_lg")

print("Loading Sentence-BERT model...")
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("All models loaded successfully!")

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
        # Suppress FastText warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    doc = nlp(text)
    alpha_tokens = [token for token in doc if token.is_alpha]
    if not alpha_tokens:
        return 0.0
    
    recognized_tokens = [token for token in alpha_tokens if not token.is_oov]
    return len(recognized_tokens) / len(alpha_tokens)

def basic_lemmatize(text):
    """
    Basic lemmatization using spaCy French model
    """
    doc = nlp(text)
    lemmatized = []
    
    for token in doc:
        if token.is_alpha:  # Only process alphabetic tokens
            lemmatized.append(token.lemma_.lower())
        else:
            lemmatized.append(token.text)
    
    return ' '.join(lemmatized)

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

def calculate_semantic_similarity(prompt, text):
    """
    Calculate semantic similarity between prompt and generated text using Sentence-BERT.
    
    Args:
        prompt (str): Original user prompt
        text (str): Generated text to compare
        
    Returns:
        float: Cosine similarity score (0.0 to 1.0)
        
    Reference:
        Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings 
        using Siamese BERT-networks. EMNLP 2019.
    """
    try:
        # Generate embeddings for both texts
        embeddings = sbert_model.encode([prompt, text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return max(0.0, float(similarity))  # Ensure non-negative
        
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0


def process_single_item(data, item_name):
    """
    Process a single JSON item (could be from a list or standalone object)
    """
    # STEP 1: Extract raw text and prompt from JSON
    raw_text = data.get('text', '') or data.get('content', '') or str(data)
    prompt = data.get('prompt', '') or data.get('input', '') or data.get('question', '')
    
    if not raw_text.strip():
        print(f"WARNING - {item_name}: No text content found")
        return
    
    # STEP 2: Clean text (remove asterisks, normalize whitespace)
    text = clean_text(raw_text)
    
    if not text.strip():
        print(f"WARNING - {item_name}: No text content after cleaning")
        return
    
    # STEP 3: Language identification using FastText
    detected_lang, lang_confidence = get_language_confidence(text)
    
    # STEP 4: Calculate French vocabulary coverage
    vocab_coverage = get_vocabulary_coverage(text)
    
    # STEP 5: Perform lemmatization
    lemmatized = basic_lemmatize(text)
    
    # STEP 6: Calculate lexical diversity (MTLD)
    mtld_score = calculate_mtld(text)
    
    # STEP 7: Calculate semantic similarity (if prompt available)
    semantic_sim = 0.0
    if prompt.strip():
        semantic_sim = calculate_semantic_similarity(prompt, text)
    
    # STEP 8: Calculate additional lexical richness metrics
    try:
        lex = LexicalRichness(text)
        ttr = lex.ttr  # Type-Token Ratio
        guiraud = lex.guiraud  # Guiraud's Index (Guiraud, 1954)
    # STEP 8: Display all scientific metrics
    except:
        ttr = 0.0
        guiraud = 0.0
    
    print(f"   Language Detection: {detected_lang} (confidence: {lang_confidence:.3f})")
    print(f"   French Vocabulary Coverage: {vocab_coverage:.3f} ({vocab_coverage*100:.1f}%)")
    print(f"   MTLD (Lexical Diversity): {mtld_score:.2f}")
    # STEP 9: Identify out-of-vocabulary tokens (degradation indicators)
        print(f"   Semantic Similarity: {semantic_sim:.3f}")
    print(f"   TTR (Type-Token Ratio): {ttr:.3f}")
    print(f"   Guiraud Index: {guiraud:.3f}")
    
    # STEP 10: Identify out-of-vocabulary tokens (degradation indicators)
    # STEP 10: Overall quality assessment based on scientific thresholds
    oov_tokens = [token.text for token in doc if token.is_alpha and token.is_oov][:5]
    if oov_tokens:
        print(f"   OOV Tokens (first 5): {', '.join(oov_tokens)}")
    
    # STEP 11: Overall quality assessment based on scientific thresholds
    if detected_lang == 'fr' and lang_confidence >= 0.9 and vocab_coverage >= 0.8:
        status = "HIGH_QUALITY"
    elif detected_lang == 'fr' and lang_confidence >= 0.7 and vocab_coverage >= 0.6:
        status = "MODERATE_QUALITY" 
    else:
        status = "LOW_QUALITY"
    print(f"   Overall Assessment: {status}")
    
    print(f"   Text sample: {text[:150]}...")
    print("-" * 80)

def process_json_files():
    """
    Process all .json files in the results folder
    """
    results_folder = "results"
    
    if not os.path.exists(results_folder):
        print(f"Error: '{results_folder}' folder not found!")
        return
    
    json_files = [f for f in os.listdir(results_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"No .json files found in '{results_folder}' folder!")
        return
    
    print(f"Processing {len(json_files)} JSON files...\n")
    
    for filename in json_files:
        filepath = os.path.join(results_folder, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Handle both single objects and lists of objects
                if isinstance(data, list):
                    # Process each item in the list
                    for i, item in enumerate(data):
                        print(f"\n=== Processing item {i+1}/{len(data)} from {filename} ===")
                        process_single_item(item, f"{filename}[{i+1}]")
                    continue
                else:
                    # Process single object
                    process_single_item(data, filename)
                    continue
                
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse {filename} as JSON")
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")

if __name__ == "__main__":
    print("French Text Analysis Tool")
    print("=" * 50)
    process_json_files()
