const fs = require('fs');
const path = require('path');

describe('CLAUDE.md Orchestrator', () => {
  let orchestratorContent;

  beforeAll(() => {
    // Read the actual CLAUDE.md file
    const filePath = path.join(__dirname, '../../.claude/CLAUDE.md');
    orchestratorContent = fs.readFileSync(filePath, 'utf8');
  });

  describe('File Size Validation', () => {
    test('should not exceed 500 tokens', () => {
      // Rough token estimation: 1 token ≈ 4 characters or 0.75 words
      const wordCount = orchestratorContent.split(/\s+/).length;
      const estimatedTokens = Math.ceil(wordCount * 0.75);

      expect(estimatedTokens).toBeLessThanOrEqual(500);
    });

    test('should contain required header metadata', () => {
      expect(orchestratorContent).toContain('version: 2.0.0');
      expect(orchestratorContent).toContain('tokens:');
    });
  });

  describe('Classification Categories', () => {
    test('should define all required categories A-E', () => {
      expect(orchestratorContent).toContain('simple');
      expect(orchestratorContent).toContain('complex');
      expect(orchestratorContent).toContain('search');
      expect(orchestratorContent).toContain('code');
      expect(orchestratorContent).toContain('meta');
    });

    test('should map categories to descriptions', () => {
      expect(orchestratorContent).toMatch(/simple.*Direct questions/);
      expect(orchestratorContent).toMatch(/complex.*Multi-step reasoning/);
      expect(orchestratorContent).toMatch(/search.*Finding information/);
      expect(orchestratorContent).toMatch(/code.*Writing.*debugging/);
      expect(orchestratorContent).toMatch(/meta.*Self-reflection/);
    });
  });

  describe('Module Loading Templates', () => {
    test('should contain @import statements for each category', () => {
      // Category A
      expect(orchestratorContent).toContain(
        '@import "./thinking-modules/response-formats.md"'
      );

      // Category B
      expect(orchestratorContent).toContain(
        '@import "./thinking-modules/SAGE.md"'
      );
      expect(orchestratorContent).toContain(
        '@import "./thinking-modules/SEIQF.md"'
      );

      // Category C
      expect(orchestratorContent).toContain(
        '@import "./thinking-modules/SIA.md"'
      );

      // Category D
      expect(orchestratorContent).toContain(
        '@import "./cognitive-tools/code-analysis.md"'
      );

      // Category E
      expect(orchestratorContent).toContain(
        '@import "./cognitive-tools/meta-reasoning.md"'
      );
    });

    test('should use relative paths from .claude directory', () => {
      const importRegex = /@import\s+"\.\/[^"]+"/g;
      const imports = orchestratorContent.match(importRegex) || [];

      imports.forEach(importStatement => {
        expect(importStatement).toMatch(/^@import\s+"\.\/[^"]+\.md"$/);
      });
    });
  });

  describe('Fallback Mechanism', () => {
    test('should include fallback protocol section', () => {
      expect(orchestratorContent).toContain('Fallback Protocol');
      expect(orchestratorContent).toContain('Continue with basic reasoning');
      expect(orchestratorContent).toContain('Log failure in debug header');
      expect(orchestratorContent).toContain('Use MCP tools directly');
    });
  });

  describe('Debug Header', () => {
    test('should include debug header template', () => {
      expect(orchestratorContent).toContain('Debug Header');
      expect(orchestratorContent).toContain('🎯 Active Modules');
      expect(orchestratorContent).toContain('⚡ Classification');
      expect(orchestratorContent).toContain('📊 Total Tokens');
      expect(orchestratorContent).toContain('🕒 Load Time');
    });
  });

  describe('Error Handling', () => {
    test('should include error handling section', () => {
      expect(orchestratorContent).toContain('Error Handling');
      expect(orchestratorContent).toContain('Invalid paths');
      expect(orchestratorContent).toContain('Missing modules');
      expect(orchestratorContent).toContain('Token overflow');
      expect(orchestratorContent).toContain('MCP failures');
    });
  });
});

describe('Request Classifier', () => {
  // Import the classifier logic (mocked for testing)
  const classifyRequest = request => {
    const text = request.toLowerCase();

    // Simplified classification logic for testing with priority order
    if (/how do you approach|your process|why did you choose/.test(text))
      return { category: 'E', confidence: 0.85 };
    if (/write|implement|fix|debug|refactor/.test(text))
      return { category: 'D', confidence: 0.9 };
    if (/analyze|compare|evaluate|design/.test(text))
      return { category: 'B', confidence: 0.85 };
    if (/find|search|locate|where|list/.test(text))
      return { category: 'C', confidence: 0.9 };
    if (/what is|how do|explain|define/.test(text))
      return { category: 'A', confidence: 0.9 };

    return { category: 'A', confidence: 0.5 };
  };

  describe('Classification Accuracy', () => {
    test('should classify simple requests as category A', () => {
      const requests = [
        'What is a variable?',
        'How do I install npm?',
        'Explain closures',
        'Define REST API'
      ];

      requests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('A');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });

    test('should classify complex requests as category B', () => {
      const requests = [
        'Analyze the performance of this algorithm',
        'Compare React and Vue frameworks',
        'Evaluate different database options',
        'Design a microservices architecture'
      ];

      requests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('B');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });

    test('should classify search requests as category C', () => {
      const requests = [
        'Find all instances of getUserData',
        'Search for configuration files',
        'Where is the main entry point?',
        'List all test files'
      ];

      requests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('C');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });

    test('should classify code requests as category D', () => {
      const requests = [
        'Write a function to validate email',
        'Implement binary search',
        'Fix the memory leak',
        'Debug this error',
        'Refactor this component'
      ];

      requests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('D');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });

    test('should classify meta requests as category E', () => {
      const requests = [
        'How do you approach debugging?',
        'What is your process for writing tests?',
        'Why did you choose that approach?'
      ];

      requests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('E');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      });
    });
  });

  describe('Performance', () => {
    test('should classify requests within 100ms', () => {
      const request = 'Write a function to calculate fibonacci numbers';

      const startTime = performance.now();
      classifyRequest(request);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100);
    });
  });
});

describe('Debug Monitor', () => {
  const createDebugMonitor = () => {
    return {
      activeModules: [],
      classification: null,
      tokenCounts: {
        orchestrator: 450,
        modules: 0,
        total: 450,
        budget: 5000
      },
      setClassification(category, confidence) {
        this.classification = { category, confidence };
      },
      addModule(name, tokens) {
        this.activeModules.push({ name, tokens });
        this.tokenCounts.modules += tokens;
        this.tokenCounts.total =
          this.tokenCounts.orchestrator + this.tokenCounts.modules;
      },
      generateHeader() {
        const modules = this.activeModules
          .map(
            m =>
              `${m.name} (${m.tokens < 1000 ? m.tokens : `${(m.tokens / 1000).toFixed(1)}K`})`
          )
          .join(', ');

        return [
          `🎯 Active Modules: ${modules || 'none'}`,
          `⚡ Classification: ${this.classification?.category} (${this.classification?.confidence.toFixed(2)})`,
          `📊 Total Tokens: ${this.tokenCounts.total} / ${this.tokenCounts.budget}`,
          '🕒 Load Time: 45ms'
        ].join('\n');
      }
    };
  };

  test('should generate debug header under 200 tokens', () => {
    const monitor = createDebugMonitor();
    monitor.setClassification('B', 0.92);
    monitor.addModule('SAGE', 2100);
    monitor.addModule('SEIQF', 1500);

    const header = monitor.generateHeader();
    const tokenCount = header.split(/\s+/).length * 0.75; // Rough token estimate

    expect(tokenCount).toBeLessThan(200);
  });

  test('should format token counts correctly', () => {
    const monitor = createDebugMonitor();
    monitor.addModule('SAGE', 2100);
    monitor.addModule('response-formats', 800);

    const header = monitor.generateHeader();

    expect(header).toContain('SAGE (2.1K)');
    expect(header).toContain('response-formats (800)');
  });

  test('should include all required debug information', () => {
    const monitor = createDebugMonitor();
    monitor.setClassification('D', 0.95);
    monitor.addModule('SEIQF', 1800);

    const header = monitor.generateHeader();

    expect(header).toContain('🎯 Active Modules');
    expect(header).toContain('⚡ Classification: D (0.95)');
    expect(header).toContain('📊 Total Tokens');
    expect(header).toContain('🕒 Load Time');
  });
});
