# Deterministic Answer RAG - Ministry of Land (MoL)

A Bengali FAQ system for Ministry of Land services using retrieval-augmented generation (RAG) with semantic similarity matching.

## Overview

This system provides automated question-answering for Ministry of Land (MoL) services in Bengali. It uses ChromaDB for vector storage, OpenAI embeddings for semantic similarity, and includes sophisticated Bengali text normalization to handle various Unicode representations.

## Features

- **Bengali Language Support**: Advanced text normalization for consistent Unicode handling
- **Semantic Search**: OpenAI embeddings with ChromaDB vector database
- **File-Based Collections**: Each FAQ file becomes a separate ChromaDB collection
- **Multiple Interfaces**: REST API, batch processing, and interactive CLI
- **Hybrid Matching**: Combines semantic similarity with exact matching

## Architecture

### Core Components

- `faq_service.py` - Main service with FAQ loading and query processing
- `bengali_normalizer.py` - Bengali text normalization and Unicode handling
- `api_server.py` - Flask REST API server
- `interactive.py` - Interactive CLI interface
- `batch_processor.py` - Batch query processing
- `faq_semantic_similarity.py` - Semantic analysis tools

### Data Structure

```
faq_data/
├── foreigner_namjari_how.txt
├── how_to_open_holding.txt
├── how_to_watch_khatian_copy.txt
├── map_fees.txt
├── namjari_mutation_fees.txt
└── ... (21 FAQ files total)
```

Each FAQ file contains question-answer pairs in Bengali, covering different aspects of land services.

## Installation

### Prerequisites
- Python 3.12+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd deterministic-answer-rag-MoL

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

### 1. Interactive CLI

```bash
python interactive.py
```

Features:
- Real-time question answering
- Debug mode with detailed matching scores
- System statistics
- Command support (`debug on/off`, `stats`, `exit`)

### 2. REST API Server

```bash
# Start server (default: localhost:5000)
python api_server.py

# Custom host/port
python api_server.py --host 0.0.0.0 --port 8080

# With debug mode
python api_server.py --debug
```

#### API Endpoints

**Single Query**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "নামজারি করতে কি ডকুমেন্ট লাগে?", "debug": false}'
```

**Batch Processing**
```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["নামজারি ফি কত?", "খতিয়ান কপি কিভাবে পাব?"]}'
```

**Health Check**
```bash
curl http://localhost:5000/api/health
```

**System Stats**
```bash
curl http://localhost:5000/api/stats
```

### 3. Batch Processor

```bash
# Process queries from file
python batch_processor.py input.txt

# Custom output file
python batch_processor.py input.txt -o results.json

# With debug information
python batch_processor.py input.txt --debug

# Show system stats
python batch_processor.py input.txt --stats
```

**Input file format** (one query per line):
```
নামজারি করার নিয়ম কি?
খতিয়ান কপি ডাউনলোড করার উপায়?
মৃত ব্যক্তির নামে ট্যাক্স দিতে হবে কি?
```

## Configuration

Edit `config.json` to customize system behavior:

```json
{
  "models": {
    "embedding_model": "text-embedding-3-small",
    "core_model": "gpt-4.1-mini"
  },
  "system": {
    "confidence_threshold": 0.0,
    "max_candidates": 1,
    "embedding_dimensions": 1024
  },
  "directories": {
    "faq_dir": "faq_data",
    "cache_dir": "cache"
  },
  "matcher_weights": {
    "exact_match": 1.0,
    "cleaned_match": 0.95,
    "ngram_match": 0.4,
    "keyword_match": 0.7,
    "embedding": 0.8,
    "boost_factor": 0.15
  }
}
```

## FAQ Data Format

Each FAQ file should contain question-answer pairs:

```
Question: নামজারি করতে কি ডকুমেন্ট প্রয়োজন?
Answer: নামজারি করতে মূল দলিল, জাতীয় পরিচয়পত্র, খতিয়ান কপি এবং অন্যান্য সংশ্লিষ্ট কাগজপত্র প্রয়োজন।

Question: নামজারি ফি কত টাকা?
Answer: নামজারি ফি জমির পরিমাণ অনুযায়ী নির্ধারিত হয়। বিস্তারিত তথ্যের জন্য সংশ্লিষ্ট অফিসে যোগাযোগ করুন।
```

## Bengali Text Processing

The system includes sophisticated Bengali text normalization:

- Unicode variation handling (ya-phala, ra-phala)
- Conjunct character normalization
- Case-insensitive matching
- Diacritical mark handling
- Consistent encoding across collections

## API Response Format

```json
{
  "query": "নামজারি করার নিয়ম কি?",
  "found": true,
  "confidence": 0.92,
  "matched_question": "নামজারি করতে কি ডকুমেন্ট প্রয়োজন?",
  "answer": "নামজারি করতে মূল দলিল, জাতীয় পরিচয়পত্র...",
  "source": "naamkharij_documents.txt",
  "collection": "faq_naamkharij_documents",
  "timestamp": "2025-08-06T12:30:15"
}
```

## Troubleshooting

### Common Issues

1. **Service not initialized**: Check FAQ files in `faq_data/` directory
2. **No OpenAI API key**: Set `OPENAI_API_KEY` environment variable
3. **Port in use**: Use different port with `--port` flag
4. **Empty results**: Ensure input files are UTF-8 encoded with proper format

### Debug Mode

Enable debug mode for detailed analysis:
- Interactive: `debug on`
- API: `"debug": true` in request
- Batch: `--debug` flag

Debug output includes:
- Collection matching details
- Similarity scores
- Candidate matches
- Processing pipeline steps

## System Statistics

- **Total Collections**: 21 (one per FAQ file)
- **Supported Languages**: Bengali (primary), English (fallback)
- **Embedding Model**: text-embedding-3-small
- **Vector Dimensions**: 1024
- **Confidence Threshold**: Configurable (default: 0.0)

## Dependencies

Core requirements:
- `openai` - OpenAI API client
- `chromadb` - Vector database
- `flask` - REST API framework
- `numpy` - Numerical operations
- `matplotlib/seaborn` - Visualization tools

See `requirements.txt` for complete dependency list.

## Development

### Adding New FAQ Collections

1. Create new `.txt` file in `faq_data/` directory
2. Follow the Question/Answer format
3. Restart the service to load new collection

### Customizing Text Processing

Modify `bengali_normalizer.py` to adjust:
- Character mappings
- Unicode normalization
- Text cleaning rules

### Tuning Similarity Matching

Adjust weights in `config.json`:
- `exact_match` - Exact string matching weight
- `embedding` - Semantic similarity weight
- `ngram_match` - N-gram matching weight
- `confidence_threshold` - Minimum match score

---

**Deterministic Answer RAG System for Ministry of Land Services**