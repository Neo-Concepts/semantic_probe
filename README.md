# Semantic Probe - A Proof of Concept

This script accompanies the NeoConcepts blog post ["Forging the Semantic Key: AI-Assisted Resilience in an Age of Information Warfare"](https://neo-concepts.com/posts/forging-the-semantic-key/).

## Purpose

`semantic_probe.py` demonstrates the technical feasibility of detecting "semantic stop signs" - weaponized words used to shut down discourse. This is an **educational tool**, not a production application.

## Quick Start

```bash
# Install dependencies
pip install nltk

# Run the script
python semantic_probe.py "Your text to analyze goes here"

# Example output
python semantic_probe.py "This woke agenda is just more authoritarian censorship"
```

## What It Does

1. **Detects Stop Signs**: Identifies potentially weaponized terms from a curated list
2. **Analyzes Context**: Examines surrounding words and sentiment
3. **Assesses Risk**: Evaluates overall manipulation likelihood

## Known Limitations

⚠️ **Important**: This tool has intentional limitations that demonstrate why human judgment is essential:

- Cannot distinguish between literal and weaponized usage ("I woke up" vs "woke agenda")
- Simple word matching without semantic understanding
- Limited to predefined stop sign list
- May flag legitimate academic or descriptive usage

These limitations are **educational features**, not bugs. They illustrate why semantic warfare detection requires human-AI collaboration, not automation.

## Example Output

```
SEMANTIC STOP SIGNS DETECTED: 2

[STOP SIGN: "woke"]
- Location: Sentence 1
- Context sentiment: Highly negative (-0.60)
- Function: Dismissal/delegitimization
- Co-occurring terms: "agenda", "more", "authoritarian"

[STOP SIGN: "authoritarian"]  
- Location: Sentence 1
- Context sentiment: Highly negative (-0.60)
- Function: Threat amplification
- Co-occurring terms: "woke", "agenda", "just", "more"

MANIPULATION RISK: HIGH
- Multiple stop signs in close proximity
- Extreme sentiment polarization
```

## Dependencies

- Python 3.7+
- nltk (automatically downloads VADER lexicon and punkt tokenizer)

## Files

- `semantic_probe.py` - Main script
- This README

## Philosophy

This tool embodies the blog post's central thesis: the semantic key to information warfare isn't an algorithm, but a methodology. The script's limitations prove why human wisdom must remain in the loop.

---

*Part of the NeoConcepts cognitive lab - exploring human-AI collaboration for better thinking.*
