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
- nltk (with vader_lexicon and punkt downloaded)
- re (built-in)
- json (built-in)

USAGE:
python semantic_probe.py "Your text to analyze goes here"
python semantic_probe.py "Your text" --human-in-loop
python semantic_probe.py "Your text" --no-adjust
python semantic_probe.py "Your text" --json

LIMITATIONS:
- The stop sign list is intentionally small and example-based
- Sentiment analysis is approximative, not definitive
- Context analysis uses simple heuristics, not deep NLP
- Results should be interpreted as indicators, not conclusions

Author: NeoConcepts / Human-AI Collaboration
License: MIT
"""

import argparse
import re
import json
from typing import List, Dict, Tuple, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

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

# Build unified data structures from SEMANTIC_STOP_SIGNS
def build_stop_sign_structures():
    """Build optimized data structures for stop sign detection."""
    term_to_category = {}
    patterns = {}
    
    for category, terms in SEMANTIC_STOP_SIGNS.items():
        for term in terms:
            term_lower = term.lower()
            term_to_category[term_lower] = category
            # Create word-boundary regex for each term
            # Handle multi-word phrases with spaces or hyphens
            if ' ' in term:
                # For multi-word phrases, allow spaces or hyphens between words
                words = [re.escape(word) for word in term.split()]
                sep = r'[\s-]+'  # allow spaces or hyphens between words
                pattern = re.compile(r'\b' + sep.join(words) + r'\b', re.IGNORECASE)
            else:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            patterns[term_lower] = pattern
    
    return term_to_category, patterns

TERM_TO_CATEGORY, PATTERNS = build_stop_sign_structures()

# Lazy-loading instances to avoid initialization race conditions
_sia_instance = None
_punkt_tokenizer_loaded = False

def ensure_nltk_resources():
    """Ensure required NLTK resources are available, downloading if needed."""
    global _punkt_tokenizer_loaded
    
    # Check and download punkt tokenizer
    if not _punkt_tokenizer_loaded:
        try:
            nltk.data.find('tokenizers/punkt')
            _punkt_tokenizer_loaded = True
        except LookupError:
            print("Downloading required NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            # Re-check after download
            try:
                nltk.data.find('tokenizers/punkt')
                _punkt_tokenizer_loaded = True
            except LookupError:
                print("Warning: Could not find punkt tokenizer after download. Sentence splitting may fail.")
    
    return _punkt_tokenizer_loaded

def get_sentiment_analyzer():
    """
    Lazily initializes and returns a singleton VADER instance.
    This ensures the lexicon is downloaded before the analyzer is created.
    """
    global _sia_instance
    if _sia_instance is None:
        try:
            _sia_instance = SentimentIntensityAnalyzer()
        except LookupError:
            print("Downloading required NLTK VADER lexicon...")
            try:
                nltk.download('vader_lexicon', quiet=True)
                _sia_instance = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Error downloading VADER lexicon: {e}")
                raise
    return _sia_instance


def preprocess_text(text: str) -> List[str]:
    """
    Tokenize text into sentences for analysis.
    
    Args:
        text: Raw input text
        
    Returns:
        List of sentences
    """
    ensure_nltk_resources()
    
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        # Fallback to simple sentence splitting if NLTK fails
        print("Warning: NLTK sentence tokenizer failed. Using simple splitting.")
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def detect_stop_signs(sentence: str) -> List[Dict[str, Any]]:
    """
    Detect ALL occurrences of semantic stop signs in a sentence using finditer.
    
    This function finds all matches (including multiple occurrences of the same term)
    and returns them sorted by position in the sentence.
    
    Args:
        sentence: A single sentence to analyze
        
    Returns:
        List of dictionaries containing detected stop signs and their positions
    """
    detected = []
    
    # Use finditer to find ALL occurrences of each term
    for term, pattern in PATTERNS.items():
        for match in pattern.finditer(sentence):
            detected.append({
                'word': match.group(),
                'term': term,  # The normalized term
                'category': TERM_TO_CATEGORY[term],
                'start_char': match.start(),
                'end_char': match.end()
            })
    
    # Sort by start position to maintain order
    detected.sort(key=lambda x: x['start_char'])
    
    return detected


def analyze_sentiment(sentence: str, adjust_for_stop_signs: bool = True) -> Tuple[float, float]:
    """
    Analyze the sentiment of a sentence using VADER with optional stop sign adjustment.
    
    VADER is particularly good at handling social media text and 
    understands intensifiers, negations, and punctuation. For our proof-of-concept,
    we optionally supplement it with semantic stop sign detection since the presence
    of weaponized language typically indicates negative intent.
    
    Args:
        sentence: Text to analyze
        adjust_for_stop_signs: Whether to apply stop sign penalty
        
    Returns:
        Tuple of (raw_sentiment, adjusted_sentiment) both in range [-1, 1]
    """
    sia = get_sentiment_analyzer()
    scores = sia.polarity_scores(sentence)
    raw_sentiment = scores['compound']
    
    if not adjust_for_stop_signs:
        return raw_sentiment, raw_sentiment
    
    # Adjust sentiment when semantic stop signs are present
    stop_signs_in_sentence = detect_stop_signs(sentence)
    
    if stop_signs_in_sentence:
        # Count unique stop sign terms (not total occurrences)
        unique_terms = set(s['term'] for s in stop_signs_in_sentence)
        
        # Apply penalty: -0.15 per unique stop sign term
        penalty = -0.15 * len(unique_terms)
        
        # Calculate adjusted sentiment
        adjusted_sentiment = raw_sentiment + penalty
        
        # If stop signs present and result is near-neutral, force slight negativity
        if adjusted_sentiment > -0.1:
            adjusted_sentiment = -0.1
        
        # Clamp to valid VADER range
        adjusted_sentiment = max(-1.0, min(1.0, adjusted_sentiment))
        
        return raw_sentiment, adjusted_sentiment
    
    return raw_sentiment, raw_sentiment


def extract_context(sentence: str, stop_sign_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract contextual information around a stop sign.
    
    This includes:
    - Co-occurring significant words
    - Detection of binary framing patterns
    - Identifying the function of the stop sign
    - Sentence snippet with highlighting
    
    Args:
        sentence: The sentence containing the stop sign
        stop_sign_info: Dictionary with stop sign details including position
        
    Returns:
        Dictionary of context analysis results
    """
    # Extract co-occurring words (exclude common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                   'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is',
                   'are', 'was', 'were', 'been', 'be', 'have', 'has',
                   'had', 'do', 'does', 'did', 'will', 'would', 'could',
                   'should', 'may', 'might', 'must', 'shall', 'can',
                   'this', 'that', 'these', 'those', 'it', 'its',
                   'we', 'they', 'you', 'as', 'if', 'than', 'not'}
    
    words = re.findall(r'\b[\w\'-]+\b', sentence.lower())
    
    # Extract tokens from the stop sign term (for multi-word phrases)
    term = stop_sign_info.get('term', '').lower()
    term_tokens = re.findall(r'\b[\w\'-]+\b', term)
    
    # Filter out common words and all tokens from the stop sign term
    co_occurring = [w for w in words 
                   if w not in common_words 
                   and w not in term_tokens
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
    function = determine_function(sentence, stop_sign_info['word'])
    
    # Create highlighted snippet
    start = stop_sign_info['start_char']
    end = stop_sign_info['end_char']
    snippet = f"{sentence[:start]}**{sentence[start:end]}**{sentence[end:]}"
    
    return {
        'co_occurring': co_occurring[:5],  # Top 5 co-occurring words
        'binary_framing': has_binary_framing,
        'function': function,
        'snippet': snippet
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


def calculate_manipulation_risk(analysis_results: List[Dict], use_adjusted: bool = True) -> Tuple[str, List[str]]:
    """
    Calculate overall manipulation risk based on analysis results.
    
    This is a simple heuristic that considers:
    - Number of stop signs detected
    - Average sentiment negativity
    - Presence of binary framing
    - Actual proximity of stop signs within sentences
    
    Args:
        analysis_results: List of analysis results for each detection
        
    Returns:
        Tuple of (risk_level, reasons)
    """
    if not analysis_results:
        return "LOW", ["No semantic stop signs detected"]
    
    reasons = []
    
    # Factor 1: Stop sign density per sentence
    sentences_with_signs = {}
    for result in analysis_results:
        sent_num = result['sentence_num']
        if sent_num not in sentences_with_signs:
            sentences_with_signs[sent_num] = 0
        sentences_with_signs[sent_num] += 1
    
    # Check for high density in any sentence
    max_density = max(sentences_with_signs.values())
    if max_density >= 3:
        reasons.append(f"High stop sign density ({max_density} in one sentence)")
    elif max_density >= 2:
        reasons.append("Multiple stop signs in same sentence")
    
    # Factor 2: Average sentiment
    sentiment_key = 'adjusted_sentiment' if use_adjusted else 'raw_sentiment'
    sentiments = [r[sentiment_key] for r in analysis_results]
    avg_sentiment = sum(sentiments) / len(sentiments)
    if avg_sentiment < -0.5:
        reasons.append("Extreme sentiment polarization")
    elif avg_sentiment < -0.2:
        reasons.append("Negative sentiment detected")
    
    # Factor 3: Binary framing
    has_binary = any(r['context']['binary_framing'] for r in analysis_results)
    if has_binary:
        reasons.append('Binary framing detected ("either...or")')
    
    # Factor 4: Total unique stop sign terms
    unique_terms = set(r['term'] for r in analysis_results)
    if len(unique_terms) >= 4:
        reasons.append(f"Multiple distinct stop sign types ({len(unique_terms)} different terms)")
    
    # Determine risk level
    risk_factors = len(reasons)
    if risk_factors >= 3 or max_density >= 3:
        risk_level = "HIGH"
    elif risk_factors >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return risk_level, reasons


def human_in_loop_confirmation(analysis_results: List[Dict]) -> List[Dict]:
    """
    Interactive human-in-the-loop confirmation for detected stop signs.
    
    This feature allows users to confirm whether each detected term is actually
    being used as semantic warfare, reinforcing that context matters.
    
    Args:
        analysis_results: List of initial analysis results
        
    Returns:
        Filtered list containing only human-confirmed weaponizations
    """
    if not analysis_results:
        return analysis_results
    
    print("\n" + "="*60)
    print("HUMAN-IN-THE-LOOP CONFIRMATION")
    print("="*60)
    print("\nPlease review each detected stop sign and confirm if it's being")
    print("used as semantic warfare (y) or legitimate usage (n):\n")
    
    confirmed_results = []
    
    for i, result in enumerate(analysis_results, 1):
        print(f"\n[{i}/{len(analysis_results)}] Detected: \"{result['word']}\"")
        print(f"Category: {result['category']}")
        print(f"Context: ...{result['context']['snippet']}...")
        print(f"Initial assessment: {result['context']['function']}")
        
        while True:
            response = input("\nIs this semantic warfare? (y/n/skip): ").lower().strip()
            if response in ['y', 'yes']:
                confirmed_results.append(result)
                print("✓ Confirmed as semantic warfare")
                break
            elif response in ['n', 'no']:
                print("✗ Marked as legitimate usage")
                break
            elif response in ['s', 'skip']:
                print("→ Skipped (excluded from analysis)")
                break
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'skip'")
    
    print("\n" + "="*60)
    print(f"Human review complete: {len(confirmed_results)}/{len(analysis_results)} confirmed")
    print("="*60 + "\n")
    
    return confirmed_results


def format_output(text: str, analysis_results: List[Dict], 
                 show_raw_sentiment: bool = False,
                 json_output: bool = False,
                 use_adjusted: bool = True) -> str:
    """
    Format the analysis results for display.
    
    Args:
        text: Original input text
        analysis_results: List of dictionaries containing analysis for each detection
        show_raw_sentiment: Whether to show raw sentiment alongside adjusted
        json_output: Whether to output JSON format
        
    Returns:
        Formatted string output
    """
    if json_output:
        # Prepare JSON-serializable output
        output_data = {
            'text': text,
            'stop_sign_count': len(analysis_results),
            'detections': []
        }
        
        for result in analysis_results:
            detection = {
                'word': result['word'],
                'category': result['category'],
                'sentence_number': result['sentence_num'],
                'position': {
                    'start': result['start_char'],
                    'end': result['end_char']
                },
                'sentiment': {
                    'raw': result['raw_sentiment'],
                    'adjusted': result['adjusted_sentiment']
                },
                'context': {
                    'function': result['context']['function'],
                    'binary_framing': result['context']['binary_framing'],
                    'co_occurring': result['context']['co_occurring'],
                    'snippet': result['context']['snippet']
                }
            }
            output_data['detections'].append(detection)
        
        risk_level, reasons = calculate_manipulation_risk(analysis_results, use_adjusted)
        output_data['risk_assessment'] = {
            'level': risk_level,
            'reasons': reasons
        }
        
        return json.dumps(output_data, indent=2)
    
    # Text output format
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
        output.append(f'- Category: {result["category"]}')
        output.append(f'- Location: Sentence {result["sentence_num"]}, chars {result["start_char"]}-{result["end_char"]}')
        
        # Format sentiment
        if show_raw_sentiment:
            raw = result['raw_sentiment']
            adjusted = result['adjusted_sentiment']
            output.append(f"- Sentiment: Raw ({raw:.2f}) → Adjusted ({adjusted:.2f})")
        else:
            sentiment = result['adjusted_sentiment'] if use_adjusted else result['raw_sentiment']
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
        
        # Show snippet
        output.append(f"- Snippet: ...{result['context']['snippet']}...")
        
        # Co-occurring terms
        co_occurring = result['context']['co_occurring']
        if co_occurring:
            co_occurring_str = '", "'.join(co_occurring)
            output.append(f'- Co-occurring terms: "{co_occurring_str}"')
        
        if i < len(analysis_results) - 1:
            output.append("")
    
    # Risk assessment
    risk_level, reasons = calculate_manipulation_risk(analysis_results, use_adjusted)
    output.append("")
    output.append(f"MANIPULATION RISK: {risk_level}")
    for reason in reasons:
        output.append(f"- {reason}")
    
    return "\n".join(output)


def analyze_text(text: str, human_confirm: bool = False, 
                adjust_for_stop_signs: bool = True,
                show_raw_sentiment: bool = False,
                json_output: bool = False) -> str:
    """
    Main analysis function that coordinates all components.
    
    Args:
        text: Input text to analyze
        human_confirm: Whether to use human-in-the-loop confirmation
        adjust_for_stop_signs: Whether to apply stop sign penalties to sentiment
        show_raw_sentiment: Whether to show raw sentiment scores
        json_output: Whether to output JSON format
        
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
            raw_sentiment, adjusted_sentiment = analyze_sentiment(
                sentence, 
                adjust_for_stop_signs=adjust_for_stop_signs
            )
            
            # Extract context
            context = extract_context(sentence, detection)
            
            # Compile results
            result = {
                'word': detection['word'],
                'term': detection['term'],
                'category': detection['category'],
                'sentence_num': sent_num,
                'sentence': sentence,
                'start_char': detection['start_char'],
                'end_char': detection['end_char'],
                'raw_sentiment': raw_sentiment,
                'adjusted_sentiment': adjusted_sentiment,
                'context': context
            }
            all_results.append(result)
    
    # Human-in-the-loop confirmation if requested
    if human_confirm and all_results:
        all_results = human_in_loop_confirmation(all_results)
    
    # Format and return output
    return format_output(text, all_results, show_raw_sentiment, json_output, adjust_for_stop_signs)


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
    
    parser.add_argument(
        '--human-in-loop', '--human',
        action='store_true',
        help='Enable human-in-the-loop confirmation of detected stop signs'
    )
    
    parser.add_argument(
        '--no-adjust',
        action='store_true',
        help='Show raw VADER sentiment without stop sign adjustment'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    # Ensure we have text to analyze
    if not args.text.strip():
        print("Error: Please provide text to analyze.")
        sys.exit(1)
    
    if args.verbose and not args.json:
        print("=== SEMANTIC PROBE ANALYSIS ===")
        print(f"Analyzing text ({len(args.text)} characters)...")
        print("=" * 30)
        print()
    
    # Perform analysis
    try:
        # Disable human-in-loop if JSON output or not interactive terminal
        import sys
        human_confirm = args.human_in_loop and not args.json and sys.stdin.isatty()
        
        output = analyze_text(
            args.text,
            human_confirm=human_confirm,
            adjust_for_stop_signs=(not args.no_adjust),
            show_raw_sentiment=args.no_adjust,
            json_output=args.json
        )
        print(output)
    except Exception as e:
        if args.json:
            error_output = json.dumps({"error": str(e)}, indent=2)
            print(error_output)
        else:
            print(f"Error during analysis: {str(e)}")
        
        if args.verbose and not args.json:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
