# Database Schema

## Module Registry (metadata.yaml)

```yaml
modules:
  sage-protocol:
    id: sage-protocol
    version: 1.0.0
    path: .claude/thinking-modules/SAGE.md
    tokenCount: 2000
    dependencies: []
    triggers:
      - 'bias'
      - 'fairness'
      - 'inclusive'
    protocols: ['SAGE']
    securityHash: 'sha256:...'
    lastModified: '2025-07-12T10:00:00Z'

  seiqf-protocol:
    id: seiqf-protocol
    version: 1.0.0
    path: .claude/thinking-modules/SEIQF.md
    tokenCount: 3000
    dependencies: []
    triggers:
      - 'search'
      - 'research'
      - 'credibility'
    protocols: ['SEIQF']
    securityHash: 'sha256:...'
    lastModified: '2025-07-12T10:00:00Z'
```

## Classification Rules (triggers.yaml)

```yaml
classifications:
  simple:
    patterns:
      - 'what is'
      - 'define'
      - 'explain briefly'
    modules: ['response-formats']
    confidence: 0.9

  complex:
    patterns:
      - 'analyze'
      - 'compare'
      - 'evaluate'
    modules: ['sage-protocol', 'cognitive-tools', 'response-formats']
    agents: ['analysis-agent']
    confidence: 0.85

  search:
    patterns:
      - 'find'
      - 'search for'
      - 'latest information'
    modules: ['sia-protocol', 'seiqf-protocol', 'response-formats']
    agents: ['research-agent']
    confidence: 0.9
```
