# Semantic Probe - Detecting Semantic Warfare Patterns

This script accompanies the NeoConcepts blog posts ["Forging the Semantic Key: AI-Assisted Resilience in an Age of Information Warfare"](https://neo-concepts.com/posts/forging-the-semantic-key/) and ["The Two Jaguars: A Case Study in Semantic Warfare vs. Business Strategy"](https://neo-concepts.com/posts/jaguar-semantic-warfare-case-study/).

## Purpose

`semantic_probe.py` demonstrates the technical feasibility of detecting semantic warfare patterns in text. Originally designed to identify "semantic stop signs," our research has evolved this into the more sophisticated "Narrative Trap" model through real-world validation.

This is an **educational tool** that reveals linguistic manipulation patterns, not a production application.

## Quick Start

```bash
# Install dependencies
pip install nltk

# Basic usage
python semantic_probe.py "Your text to analyze goes here"

# Human-in-the-loop confirmation
python semantic_probe.py "This woke agenda is just virtue signaling" --human-in-loop

# Show raw VADER sentiment without stop sign adjustment
python semantic_probe.py "Stop being such a snowflake" --no-adjust

# JSON output for downstream processing
python semantic_probe.py "Echo chamber of radical extremists" --json

# Verbose mode for debugging
python semantic_probe.py "Cancel culture strikes again" --verbose
```

## What It Does

1. **Detects Semantic Patterns**: Identifies ALL occurrences of potentially weaponized terms (including repeated usage)
2. **Analyzes Context**: Examines surrounding words, sentiment, and rhetorical function with enhanced accuracy
3. **Reveals Manipulation Techniques**: Shows categorical shutdown, threat amplification, and binary framing patterns
4. **Assesses Risk**: Evaluates manipulation likelihood using density analysis and sentiment polarization
5. **Human-in-the-Loop**: Optional interactive confirmation to distinguish weaponization from legitimate usage

## Conceptual Evolution

**Semantic Stop Signs** → **Narrative Traps**

Our analysis of real-world semantic warfare (like the Jaguar rebrand controversy) revealed that weaponized language doesn't just end conversations—it creates interpretive frameworks that predetermine how all future evidence will be understood. This tool helps identify both phenomena.

## Known Limitations

⚠️ **Important**: This tool has intentional limitations that demonstrate why human judgment is essential:

- Cannot distinguish between literal and weaponized usage ("I woke up" vs "woke agenda")
- Simple word matching without semantic understanding
- Limited to predefined stop sign list
- May flag legitimate academic or descriptive usage

These limitations are **educational features**, not bugs. They illustrate why semantic warfare detection requires human-AI collaboration, not automation.

## Version History

The current version incorporates critical fixes identified through rigorous code review:
- Fixed NLTK resource handling for reliability
- Improved detection accuracy with `finditer` for multiple occurrences
- Added human-in-the-loop confirmation for pedagogical value
- Enhanced sentiment analysis with proper clamping
- Better context extraction excluding multi-word phrase tokens

## Key Features

- **Accurate Multiple Detection**: Uses `finditer` to catch ALL occurrences, even repeated words
- **Flexible Phrase Matching**: Detects both "virtue signaling" and "virtue-signaling" 
- **Human-in-the-Loop Confirmation**: Interactive mode to verify weaponization vs legitimate usage
- **Raw Sentiment Display**: `--no-adjust` flag shows VADER baseline without stop sign penalties
- **JSON Export**: Machine-readable output with snippets for downstream processing
- **Enhanced Context**: Character-level position tracking and highlighted snippets
- **Smart Co-occurrence**: Properly excludes multi-word phrase tokens from analysis
- **Density-Based Risk**: Analyzes stop sign concentration per sentence, not just total count

## Risk Assessment Levels Explained

The tool calculates manipulation risk based on multiple factors:

### Risk Levels

**LOW Risk** (0-1 factors present):
- Isolated usage of loaded terms
- Single factor like negative sentiment
- Minimal manipulation tactics

**MEDIUM Risk** (2 factors present):
- Clear semantic warfare patterns emerging
- Multiple tactics working together
- Strategic use of language weapons

**HIGH Risk** (3+ factors OR high density):
- Full semantic trap deployment
- Concentrated weaponization to shut down discourse
- Multiple manipulation techniques combined
- 3+ stop signs in a single sentence (automatic HIGH)

### Risk Factors Evaluated

1. **Stop Sign Density**: Multiple stop signs in the same sentence
2. **Sentiment Polarization**: Extreme negative sentiment (< -0.5)
3. **Binary Framing**: "Either...or" patterns that force false choices
4. **Term Variety**: 4+ different types of stop signs across text


## Example Output

```
SEMANTIC STOP SIGNS DETECTED: 2

[STOP SIGN: "woke"]
- Category: dismissive_labels
- Location: Sentence 1, chars 5-9
- Context sentiment: Highly negative (-0.60)
- Function: Dismissal/delegitimization
- Snippet: ...This **woke** agenda is just more authoritarian censorship...
- Co-occurring terms: "agenda", "just", "more", "censorship"

[STOP SIGN: "authoritarian"]  
- Category: ideological_accusations
- Location: Sentence 1, chars 30-43
- Context sentiment: Highly negative (-0.60)
- Function: Threat amplification
- Snippet: ...This woke agenda is just more **authoritarian** censorship...
- Co-occurring terms: "agenda", "just", "more", "censorship"

MANIPULATION RISK: HIGH
- Multiple stop signs in same sentence
- Extreme sentiment polarization
```

### JSON Output Example

```json
{
  "text": "Stop being such a snowflake.",
  "stop_sign_count": 1,
  "detections": [{
    "word": "snowflake",
    "category": "dismissive_labels",
    "sentence_number": 1,
    "position": {"start": 18, "end": 27},
    "sentiment": {"raw": -0.296, "adjusted": -0.446},
    "context": {
      "function": "Categorical shutdown",
      "binary_framing": false,
      "co_occurring": ["stop", "being", "such"],
      "snippet": "Stop being such a **snowflake**."
    }
  }],
  "risk_assessment": {
    "level": "LOW",
    "reasons": ["Negative sentiment detected"]
  }
}
```

## Dependencies

- Python 3.7+
- nltk (automatically downloads VADER lexicon and punkt tokenizer)

## Command-Line Options

- `text` - The text to analyze (required, enclose in quotes)
- `--human-in-loop`, `--human` - Enable interactive confirmation of detected stop signs
- `--no-adjust` - Show raw VADER sentiment without stop sign penalties
- `--json` - Output results in JSON format for downstream processing
- `--verbose`, `-v` - Show additional debug information

## Files

- `semantic_probe.py` - Main script with all enhancements
- `semantic_probe_v1_backup.py` - Original version (kept for reference)
- This README

## Real-World Validation

The tool was validated through analysis of the 2024 Jaguar rebrand controversy, where it successfully identified how the "woke" label created a narrative trap that captured all subsequent business data (sales declines, leadership changes) and forced it to be interpreted as proof of the original semantic attack.

## Philosophy

This tool embodies the NeoConcepts thesis: real thinking tools don't just organize ideas—they force uncomfortable reckonings with reality. The semantic key to information warfare isn't an algorithm, but a methodology that combines pattern detection with human critical thinking.

The script's limitations are educational features that demonstrate why semantic warfare detection requires human-AI collaboration, not automation.

---

*Part of the NeoConcepts cognitive lab - exploring human-AI collaboration for better thinking.*
