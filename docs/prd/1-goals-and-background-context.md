# 1. Goals and Background Context

## Goals
- Reduce Claude Code context window usage from 38K tokens to 2-5K tokens per request
- Enable universal thinking capabilities without explicit command triggers
- Create maintainable, modular prompt architecture that's easy to update
- Integrate seamlessly with clear-thought MCP for thinking mechanisms
- Preserve all existing SAGE, SEIQF, SIA functionality while improving performance
- Enable thinking to enhance every interaction automatically based on context
- Support Universal Dynamic Information Gathering where thinking tools can invoke other tools mid-execution
- Achieve 90%+ user satisfaction rating compared to CLAUDE-v3.md baseline
- Maintain 100% feature parity with existing system while improving performance

## Background Context
The universal-claude-thinking project addresses critical limitations in current LLM prompt engineering. As we add features to simulate human-like thinking and prevent biases, context windows grow unsustainably large. The existing CLAUDE-v3.md file at 38,221 tokens exceeds practical limits, causing quality degradation and maintenance challenges. 

Research from leading repositories (Claude-Code-Development-Kit, SuperClaude, Context-Engineering) and academic papers (IBM Zurich's cognitive tools, ICML's emergent symbolic mechanisms) shows that modular, dynamically-loaded architectures can maintain functionality while dramatically reducing context usage. This project implements these best practices to create an efficient, universal thinking layer for Claude Code.

## Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-07-12 | 1.0.0 | Initial PRD creation | John (PM) |
| 2025-07-12 | 1.1.0 | Added critical missing stories per PO validation | Sarah (PO) |
