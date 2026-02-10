# Aegis Insight v1.2.0 Release Notes

**Release Date:** February 2026  
**Codename:** Unified Enrichment

---

## Overview

Aegis Insight v1.2.0 delivers significant performance improvements to the document extraction pipeline, reducing processing time by **25-30%** through intelligent prompt consolidation. This release also includes critical bug fixes for JSON parsing and introduces a powerful trace mode for debugging extraction issues.

---

## What's New

### 🚀 Mega-Prompt Enrichment (25-30% Faster)

The extraction pipeline now consolidates five separate LLM calls into a single unified prompt for claim enrichment. Previously, each claim batch required separate calls for:

- Temporal extraction (dates, time references)
- Geographic extraction (locations, cultural context)
- Citation analysis (source attribution)
- Emotion detection (sentiment, manipulation signals)
- Authority assessment (domain expertise, credibility)

**Before:** 5 sequential LLM calls × ~5 seconds each = ~25 seconds per chunk  
**After:** 1 unified LLM call = ~12-16 seconds per chunk

This optimization benefits all users regardless of hardware configuration.

#### Automatic Fallback

If the mega-prompt fails to parse correctly, the system automatically falls back to individual extractors — ensuring no data loss while maximizing throughput when conditions allow.

---

### 🔧 Improved JSON Parser

Fixed critical parsing issues that caused extraction failures:

- **Markdown Code Fence Handling:** LLM responses wrapped in ` ```json ` blocks are now properly stripped before parsing
- **Bracket Balancing:** New parser correctly extracts JSON arrays even when the LLM includes explanation text after the array
- **Greedy Regex Fix:** Parser no longer captures unrelated `[1]`, `[2]` markers in explanation text

**Impact:** Geographic and citation extractors now succeed where they previously returned empty defaults.

---

### 🔍 Trace Mode for Debugging

New environment variable `AEGIS_TRACE=true` enables full visibility into the extraction pipeline:
```bash
AEGIS_TRACE=true python3 run_extraction_pipeline_v3.py \
  --jsonl input.jsonl \
  --job-id debug_test \
  --max-chunks 5
```

Trace mode logs:
- Complete prompts sent to the LLM
- Raw LLM responses (with timing)
- JSON extraction attempts
- Parse success/failure with context

This is invaluable for:
- Debugging extraction failures on specific document types
- Tuning prompts for specialized corpora
- Understanding model behavior with unusual content

---

## Performance Comparison

| Metric | v1.1.x | v1.2.0 | Improvement |
|--------|--------|--------|-------------|
| Enrichment time (5 claims) | ~25s | ~14s | **44% faster** |
| Total chunk processing | ~45s | ~35s | **22% faster** |
| LLM calls per chunk | 5 | 1 | **80% reduction** |
| Full corpus processing | ~55 days* | ~38 days* | **31% faster** |

*Based on 294K chunks on dual RTX A6000 GPUs

---

## Upgrade Instructions

### Docker Users
```bash
# Pull latest image
docker pull aegisinsight/aegis-insight:1.2.0

# Or update docker-compose.yml
services:
  aegis:
    image: aegisinsight/aegis-insight:1.2.0
```

### Source Installation
```bash
# Pull latest changes
git pull origin main

# The extraction orchestrator is automatically updated
# No database migrations required
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AEGIS_TRACE` | `false` | Enable full prompt/response logging |
| `OLLAMA_HOST` | `http://localhost:11434` | Primary Ollama endpoint |
| `OLLAMA_SECONDARY_URL` | `http://localhost:11435` | Secondary Ollama for dual-GPU |

### Command Line
```bash
# Standard extraction
python3 run_extraction_pipeline_v3.py --jsonl corpus.jsonl --job-id my_job

# Resume interrupted job
python3 run_extraction_pipeline_v3.py --jsonl corpus.jsonl --job-id my_job --resume

# Debug mode with trace
AEGIS_TRACE=true python3 run_extraction_pipeline_v3.py --jsonl corpus.jsonl --job-id debug --max-chunks 10
```

---

## Technical Details

### Mega-Prompt Structure

The unified prompt requests all five dimensions in a single response:
```json
[
  {
    "temporal": {"absolute_dates": [], "relative_dates": []},
    "geographic": {"locations": [], "cultural_context": []},
    "citation": {"cites_other_work": false, "attribution_chain": []},
    "emotion": {"primary_sentiment": "neutral", "emotional_intensity": 0.0, ...},
    "authority": {"claim_domain": "other", "domain_match": 0.5, ...}
  }
]
```

### JSON Extraction Algorithm

The new `_extract_json_array()` method uses bracket balancing instead of regex:

1. Find first `[` character
2. Track bracket depth, respecting string escaping
3. Return substring when brackets balance
4. Fall back to greedy match if balancing fails

This handles edge cases where LLMs include:
- Explanation text after the JSON
- Reference markers like `[1]`, `[2]` in prose
- Incomplete responses (graceful degradation)

---

## Known Limitations

- **Very large claim batches (10+ claims):** Mega-prompt responses can be slow; individual extractors may be faster in edge cases. Fallback handles this automatically.
- **Non-standard Ollama deployments:** Trace mode assumes standard logging; custom log handlers may need adjustment.

---

## What's Next

### v1.3.0 Roadmap

- **Parallel enrichment for multi-GPU:** Run different extractors on different GPUs simultaneously
- **Adaptive batching:** Dynamically adjust claim batch sizes based on content complexity
- **Streaming extraction:** Begin parsing before full LLM response completes

---

## Acknowledgments

This release incorporates feedback from early adopters processing large document corpora. Special thanks to the community for identifying edge cases in JSON parsing and suggesting the trace mode feature.

---

## Links

- **Documentation:** https://docs.aegisinsight.ai
- **GitHub:** https://github.com/aegis-insight/aegis-insight
- **Discord:** https://discord.gg/aegisinsight
- **Issues:** https://github.com/aegis-insight/aegis-insight/issues

---

*Aegis Insight — Epistemic Defense Infrastructure*
