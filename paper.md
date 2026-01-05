---
title: 'Aegis Insight: Knowledge Graph Infrastructure for Detecting Suppression and Coordination Patterns in Document Corpora'
tags:
  - Python
  - knowledge graphs
  - information retrieval
  - misinformation detection
  - natural language processing
  - RAG systems
authors:
  - name: Robert Beken
    orcid: 0009-0003-0955-7198
    corresponding: true
    affiliation: 1
affiliations:
  - name: Cedrus Strategic LLC, United States
    index: 1
date: 3 January 2026
bibliography: paper.bib
---

# Summary

Aegis Insight is an open-source knowledge graph system that detects suppression and coordination patterns in document corpora. Unlike misinformation detection systems that identify false claims, Aegis Insight addresses fundamentally different questions: whether true claims are systematically suppressed and whether consensus around accurate information is artificially manufactured.

The system employs a seven-dimensional extraction pipeline that processes documents into typed claims with entities, temporal markers, geographic references, citations, emotional content, and authority-domain relationships. Detection algorithms implement threshold-based "Goldfinger" scoring where isolated indicators score minimally but accumulated indicators trigger exponential escalation. The system runs entirely locally on consumer hardware via Docker, using local LLM inference through Ollama with no cloud dependencies.

# Statement of Need

Current retrieval-augmented generation (RAG) systems optimize for information retrieval accuracy while remaining blind to the epistemological structure of information landscapes [@Lewis2020; @Gao2023]. These systems operate under an implicit assumption that the information landscape itself is epistemologically sound—that high-quality research receives proportional visibility and that consensus emerges organically from evidence evaluation.

This assumption fails under documented conditions including systematic suppression of research through institutional gatekeeping, and manufactured consensus through coordinated messaging across nominally independent sources. Historical cases provide unambiguous ground truth: Thomas Paine's prosecution and posthumous erasure [@Fruchtman1994], the Business Plot suppression despite Congressional verification [@Archer2007], and Yellow Journalism's coordinated campaign preceding the Spanish-American War [@Campbell2001].

Existing misinformation detection research focuses on identifying false claims rather than detecting suppression of true claims [@Shu2017; @Zhou2020]. Recent work on news omission [@Horne2019] and cross-document event graphs [@Wu2022] extends in this direction, but no existing tool provides integrated detection of suppression signatures and coordination patterns suitable for non-expert users.

Aegis Insight fills this gap by providing:

- **Suppression Detection**: Identifies quality-visibility gaps, network isolation, and institutional dismissal patterns through semantic analysis of corpus text
- **Coordination Detection**: Detects temporal clustering, language similarity, and synchronized emotional triggers across sources
- **Anomaly Detection**: Finds cross-cultural patterns that deviate from expected baselines
- **Local Deployment**: Runs entirely on consumer hardware with Docker, requiring no cloud dependencies

Validation against historical ground truth demonstrates effective discrimination between documented suppression cases (Thomas Paine: 0.832, Smedley Butler: 0.784) and appropriate controls (Benjamin Franklin: 0.394), with coordination detection correctly identifying the Yellow Journalism campaign of 1898.

The system is designed for integration with existing RAG systems via Model Context Protocol (MCP) endpoints, enabling AI assistants to acknowledge epistemological structure in their responses. Aegis Insight serves researchers, journalists, and citizens seeking to understand not just what information exists but what structural forces shape its visibility.

# Acknowledgements

The author thanks the open-source communities behind Neo4j, PostgreSQL, Ollama, and the Mistral model family for providing the foundational infrastructure that makes local deployment possible.

# References
