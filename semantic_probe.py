#!/usr/bin/env python3
"""
semantic_probe.py - A Proof-of-Concept for Detecting Semantic Stop Signs

PURPOSE:
This script demonstrates the technical feasibility of detecting and analyzing 
"semantic stop signs" - weaponized words used to shut down discourse. It's an 
educational tool accompanying the NeoConcepts post "Forging the Semantic Key."

CONCEPTUAL EVOLUTION:
While this tool detects 'semantic stop signs,' our broader research has evolved 
this concept into the more sophisticated 'Narrative Trap' model. Semantic stop 
signs end conversations; narrative traps control the interpretation of ongoing 
events by creating frameworks that predetermine how all future evidence will be 
understood. This tool helps evidence the mechanics of both phenomena.

WHAT THIS IS:
- A simple demonstration of AI-assisted semantic analysis
- An educational tool showing how detection might work
- A starting point for understanding linguistic manipulation
- A proof-of-concept for the Narrative Trap model validation

WHAT THIS IS NOT:
- A production-ready tool
- A comprehensive detector of all semantic weapons
- A replacement for human critical thinking

DEPENDENCIES:
- Python 3.7+
- nltk (with vader_lexicon downloaded)
- re (built-in)

USAGE:
python semantic_probe.py "Your text to analyze goes here"

LIMITATIONS:
- The stop sign list is intentionally small and example-based
- Sentiment analysis is approximative, not definitive
- Context analysis uses simple heuristics, not deep NLP
- Results should be interpreted as indicators, not conclusions

Author: NeoConcepts / Human-AI Collaboration
Website: www.neo-concepts.com
License: MIT
"""

import argparse
import re
import sys
from typing import List, Dict, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('punkt_tab')


# Curated list of semantic stop signs grouped by function
# 
# IMPORTANT LIMITATION: This simple word-matching approach cannot distinguish
# between weaponized and legitimate usage of the same word. For example:
# - "I woke up early" (literal) vs "that's so woke" (weaponized)
# - "academic privilege" (legitimate) vs "check your privilege" (dismissive)
# 
# This limitation is INTENTIONAL for educational purposes - it demonstrates
# why semantic warfare detection requires human judgment, not just algorithms.
SEMANTIC_STOP_SIGNS = {
    'dismissive_labels': [
        'woke', 'snowflake', 'boomer', 'karen', 'simp'
    ],
    'ideological_accusations': [
        'fascist', 'communist', 'socialist', 'authoritarian', 
        'extremist', 'radical'
    ],
    'conceptual_weaponization': [
        'censorship', 'cancel', 'cancelled', 'virtue signaling',
        'echo chamber', 'safe space'
    ],
    'binary_forcers': [
        'sheep', 'sheeple', 'npc', 'shill', 'bootlicker'
    ]
}

# Flatten the dictionary into a single list for easier searching
ALL_STOP_SIGNS = []
STOP_SIGN_CATEGORIES = {}
for category, words in SEMANTIC_STOP_SIGNS.items():
    for word in words:
        ALL_STOP_SIGNS.append(word.lower())
        STOP_SIGN_CATEGORIES[word.lower()] = category


# Lazy-loading sentiment analyzer to avoid initialization race condition
_sia_instance = None
def get_sentiment_analyzer():
    """
    Lazily initializes and returns a singleton VADER instance.
    This ensures the lexicon is downloaded before the analyzer is created.
    """
    global _sia_instance
    if _sia_instance is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _sia_instance = SentimentIntensityAnalyzer()
    return _sia_instance


def preprocess_text(text: str) -> List[str]:
    """
    Tokenize text into sentences for analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        List of sentences
    """
    # Use NLTK's sentence tokenizer for better accuracy
    sentences = nltk.sent_tokenize(text)
    return sentences


def detect_stop_signs(sentence: str) -> List[Dict[str, str]]:
    """
    Detect semantic stop signs in a sentence.
    
    This function looks for exact word matches (case-insensitive) rather than
    substring matches to avoid false positives (e.g., "awoken" matching "woke").
    
    Args:
        sentence: A single sentence to analyze
        
    Returns:
        List of dictionaries containing detected stop signs and their positions
    """
    detected = []
    sentence_lower = sentence.lower()
    words = re.findall(r'\b[\w\'-]+\b', sentence_lower)
    
    for i, word in enumerate(words):
        if word in ALL_STOP_SIGNS:
            # Find the actual position in the original sentence
            # This preserves the original casing for display
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                detected.append({
                    'word': match.group(),
                    'category': STOP_SIGN_CATEGORIES[word],
                    'position': i,
                    'start_char': match.start(),
                    'end_char': match.end()
                })
    
    # Check for multi-word phrases
    for phrase in ['virtue signaling', 'echo chamber', 'safe space']:
        if phrase in sentence_lower:
            match = re.search(r'\b' + re.escape(phrase) + r'\b', 
                            sentence, re.IGNORECASE)
            if match:
                detected.append({
                    'word': match.group(),
                    'category': STOP_SIGN_CATEGORIES[phrase.replace(' ', ' ')],
                    'position': -1,  # Multi-word phrase
                    'start_char': match.start(),
                    'end_char': match.end()
                })
    
    return detected


def analyze_sentiment(sentence: str) -> float:
    """
    Analyze the sentiment of a sentence using VADER with semantic stop sign adjustment.
    
    VADER is particularly good at handling social media text and 
    understands intensifiers, negations, and punctuation. However, for 
    our proof-of-concept, we supplement it with semantic stop sign detection
    since the presence of weaponized language typically indicates negative intent.
    
    Args:
        sentence: Text to analyze
        
    Returns:
        Compound sentiment score (-1 to 1)
    """
    sia = get_sentiment_analyzer()  # Get the initialized instance
    scores = sia.polarity_scores(sentence)
    base_sentiment = scores['compound']
    
    # For proof-of-concept: adjust sentiment when semantic stop signs are present
    # This makes the demo more realistic since weaponized language often carries
    # negative intent even when VADER doesn't detect it
    stop_signs_in_sentence = detect_stop_signs(sentence)
    if stop_signs_in_sentence:
        # Each stop sign adds negative weight
        stop_sign_penalty = len(stop_signs_in_sentence) * -0.3
        adjusted_sentiment = min(base_sentiment + stop_sign_penalty, -0.1)
        return adjusted_sentiment
    
    return base_sentiment


def extract_context(sentence: str, stop_sign: str) -> Dict[str, any]:
    """
    Extract contextual information around a stop sign.
    
    This includes:
    - Co-occurring significant words
    - Detection of binary framing patterns
    - Identifying the function of the stop sign
    
    Args:
        sentence: The sentence containing the stop sign
        stop_sign: The detected stop sign word
        
    Returns:
        Dictionary of context analysis results
    """
    # Extract co-occurring words (exclude common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                   'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is',
                   'are', 'was', 'were', 'been', 'be', 'have', 'has',
                   'had', 'do', 'does', 'did', 'will', 'would', 'could',
                   'should', 'may', 'might', 'must', 'shall', 'can',
                   'this', 'that', 'these', 'those', 'it', 'its'}
    
    words = re.findall(r'\b[\w\'-]+\b', sentence.lower())
    co_occurring = [w for w in words 
                   if w not in common_words 
                   and w != stop_sign.lower()
                   and len(w) > 2]
    
    # Detect binary framing patterns
    binary_patterns = [
        r'\beither\b.*\bor\b',
        r'\bif you\'re not\b.*\byou\'re\b',
        r'\bonly\b.*\bwould\b',
        r'\banyone who\b.*\bis\b',
        r'\byou\'re either\b.*\bor\b'
    ]
    
    has_binary_framing = any(re.search(pattern, sentence, re.IGNORECASE) 
                            for pattern in binary_patterns)
    
    # Determine function based on context
    function = determine_function(sentence, stop_sign)
    
    return {
        'co_occurring': co_occurring[:5],  # Top 5 co-occurring words
        'binary_framing': has_binary_framing,
        'function': function
    }


def determine_function(sentence: str, stop_sign: str) -> str:
    """
    Determine the rhetorical function of a stop sign in context.
    
    This is a simple heuristic based on surrounding words and patterns.
    
    Args:
        sentence: The sentence containing the stop sign
        stop_sign: The detected stop sign
        
    Returns:
        String describing the likely function
    """
    sentence_lower = sentence.lower()
    
    # Check for dismissal patterns
    dismissal_patterns = ['just another', 'typical', 'nothing but', 
                         'same old', 'predictable']
    if any(pattern in sentence_lower for pattern in dismissal_patterns):
        return "Dismissal/delegitimization"
    
    # Check for threat amplification
    threat_patterns = ['destroy', 'attack', 'danger', 'threat', 
                      'undermine', 'resist', 'fight']
    if any(pattern in sentence_lower for pattern in threat_patterns):
        return "Threat amplification"
    
    # Check for identity attacks
    identity_patterns = ['you are', 'they are', 'you\'re', 'they\'re',
                        'sounds like', 'typical of']
    if any(pattern in sentence_lower for pattern in identity_patterns):
        return "Identity-based dismissal"
    
    # Default to general categorization
    return "Categorical shutdown"


def calculate_manipulation_risk(analysis_results: List[Dict]) -> Tuple[str, List[str]]:
    """
    Calculate overall manipulation risk based on analysis results.
    
    This is a simple heuristic that considers:
    - Number of stop signs detected
    - Average sentiment negativity
    - Presence of binary framing
    
    Args:
        analysis_results: List of analysis results for each detection
        
    Returns:
        Tuple of (risk_level, reasons)
    """
    if not analysis_results:
        return "LOW", ["No semantic stop signs detected"]
    
    reasons = []
    
    # Factor 1: Number of stop signs
    stop_sign_count = len(analysis_results)
    if stop_sign_count >= 2:
        reasons.append("Multiple stop signs in close proximity")
    
    # Factor 2: Average sentiment
    sentiments = [r['sentiment'] for r in analysis_results]
    avg_sentiment = sum(sentiments) / len(sentiments)
    if avg_sentiment < -0.5:
        reasons.append("Extreme sentiment polarization")
    elif avg_sentiment < -0.2:
        reasons.append("Negative sentiment detected")
    
    # Factor 3: Binary framing
    has_binary = any(r['context']['binary_framing'] for r in analysis_results)
    if has_binary:
        reasons.append('Binary framing detected ("either...or")')
    
    # Determine risk level
    risk_factors = len(reasons)
    if risk_factors >= 3:
        risk_level = "HIGH"
    elif risk_factors >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return risk_level, reasons


def format_output(text: str, analysis_results: List[Dict]) -> str:
    """
    Format the analysis results to match the blog post example.
    
    Args:
        text: Original input text
        analysis_results: List of dictionaries containing analysis for each detection
        
    Returns:
        Formatted string output
    """
    output = []
    
    # Header
    stop_sign_count = len(analysis_results)
    output.append(f"SEMANTIC STOP SIGNS DETECTED: {stop_sign_count}")
    
    if stop_sign_count == 0:
        output.append("\nNo semantic stop signs detected in the input text.")
        output.append("\nMANIPULATION RISK: LOW")
        return "\n".join(output)
    
    output.append("")
    
    # Individual stop sign analysis
    for i, result in enumerate(analysis_results):
        output.append(f'[STOP SIGN: "{result["word"]}"]')
        output.append(f'- Location: Sentence {result["sentence_num"]}')
        
        # Format sentiment
        sentiment = result['sentiment']
        if sentiment < -0.5:
            sentiment_desc = f"Highly negative ({sentiment:.2f})"
        elif sentiment < -0.1:
            sentiment_desc = f"Negative ({sentiment:.2f})"
        elif sentiment > 0.5:
            sentiment_desc = f"Highly positive ({sentiment:.2f})"
        elif sentiment > 0.1:
            sentiment_desc = f"Positive ({sentiment:.2f})"
        else:
            sentiment_desc = f"Neutral ({sentiment:.2f})"
        
        output.append(f"- Context sentiment: {sentiment_desc}")
        output.append(f"- Function: {result['context']['function']}")
        
        # Co-occurring terms
        co_occurring = result['context']['co_occurring']
        if co_occurring:
            co_occurring_str = '", "'.join(co_occurring)
            output.append(f'- Co-occurring terms: "{co_occurring_str}"')
        
        if i < len(analysis_results) - 1:
            output.append("")
    
    # Risk assessment
    risk_level, reasons = calculate_manipulation_risk(analysis_results)
    output.append("")
    output.append(f"MANIPULATION RISK: {risk_level}")
    for reason in reasons:
        output.append(f"- {reason}")
    
    return "\n".join(output)


def analyze_text(text: str) -> str:
    """
    Main analysis function that coordinates all components.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Formatted analysis output
    """
    # Preprocess text into sentences
    sentences = preprocess_text(text)
    
    # Analyze each sentence
    all_results = []
    
    for sent_num, sentence in enumerate(sentences, 1):
        # Detect stop signs in this sentence
        detections = detect_stop_signs(sentence)
        
        for detection in detections:
            # Analyze sentiment of the sentence
            sentiment = analyze_sentiment(sentence)
            
            # Extract context
            context = extract_context(sentence, detection['word'])
            
            # Compile results
            result = {
                'word': detection['word'],
                'category': detection['category'],
                'sentence_num': sent_num,
                'sentence': sentence,
                'sentiment': sentiment,
                'context': context
            }
            all_results.append(result)
    
    # Format and return output
    return format_output(text, all_results)


def main():
    """
    Command-line interface for the semantic probe.
    """
    parser = argparse.ArgumentParser(
        description='Detect and analyze semantic stop signs in text.',
        epilog='Example: python semantic_probe.py "Your text to analyze"'
    )
    
    parser.add_argument(
        'text',
        help='The text to analyze (enclose in quotes)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show additional debug information'
    )
    
    args = parser.parse_args()
    
    # Ensure we have text to analyze
    if not args.text.strip():
        print("Error: Please provide text to analyze.")
        sys.exit(1)
    
    if args.verbose:
        print("=== SEMANTIC PROBE ANALYSIS ===")
        print(f"Analyzing text ({len(args.text)} characters)...")
        print("=" * 30)
        print()
    
    # Perform analysis
    try:
        output = analyze_text(args.text)
        print(output)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
