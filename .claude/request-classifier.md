# Request Classifier

## Classification Logic

### Category Mappings
- **A/simple**: Direct questions, basic explanations
- **B/complex**: Multi-step reasoning, analysis tasks  
- **C/search**: Finding information, code exploration
- **D/code**: Writing, refactoring, debugging code
- **E/meta**: Self-reflection, process questions

### Classification Function

```javascript
function classifyRequest(request) {
  const text = request.toLowerCase();
  const wordCount = text.split(/\s+/).length;
  
  const patterns = {
    A: {
      patterns: [/what is/i, /how do/i, /explain/i, /define/i],
      maxWords: 50,
      confidence: 0.9
    },
    B: {
      patterns: [/analyze/i, /compare/i, /evaluate/i, /design/i],
      indicators: ['multiple', 'steps', 'consider'],
      confidence: 0.85
    },
    C: {
      patterns: [/find/i, /search/i, /locate/i, /where/i, /list/i],
      fileIndicators: ['.js', '.md', 'file', 'directory'],
      confidence: 0.9
    },
    D: {
      patterns: [/write/i, /implement/i, /fix/i, /debug/i, /refactor/i],
      codeIndicators: ['function', 'class', 'bug', 'error', 'code'],
      confidence: 0.9
    },
    E: {
      patterns: [/how do you/i, /your process/i, /why did you/i],
      selfReference: ['you', 'your', 'claude'],
      confidence: 0.85
    }
  };
  
  let bestMatch = { category: 'A', confidence: 0.5 };
  
  for (const [category, config] of Object.entries(patterns)) {
    let score = 0;
    let matches = 0;
    
    config.patterns?.forEach(pattern => {
      if (pattern.test(text)) {
        score += config.confidence;
        matches++;
      }
    });
    
    config.indicators?.forEach(indicator => {
      if (text.includes(indicator)) {
        score += 0.2;
        matches++;
      }
    });
    
    if (matches > 0) {
      const avgScore = score / matches;
      if (avgScore > bestMatch.confidence) {
        bestMatch = { category, confidence: avgScore };
      }
    }
  }
  
  return bestMatch;
}

// Module mapping
const moduleMapping = {
  A: ['response-formats'],
  B: ['SAGE', 'SEIQF', 'cognitive-tools/analysis'],
  C: ['SIA', 'cognitive-tools/search'],
  D: ['SEIQF', 'cognitive-tools/code-analysis'],
  E: ['SAGE', 'cognitive-tools/meta-reasoning']
};

function getRequiredModules(category) {
  return moduleMapping[category] || moduleMapping.A;
}
```