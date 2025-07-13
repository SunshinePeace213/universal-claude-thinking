const { 
  classifyRequest, 
  getRequiredModules, 
  getModulePriority,
  LRUCache,
  TelemetryLogger,
  generateSecureCacheKey,
  CONFIDENCE_WEIGHTS,
  CACHE_CONFIG,
  TELEMETRY_CONFIG
} = require('../../.claude/request-classifier.md');

describe('Request Classifier', () => {
  describe('classifyRequest', () => {
    test('should classify simple requests correctly', () => {
      const simpleRequests = [
        'What is JavaScript?',
        'How do I install npm?',
        'Explain closures in JS',
        'Define REST API'
      ];

      simpleRequests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('simple');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
        expect(result.requiredModules).toContain('response-formats');
        expect(result.suggestedAgents).toContain('responder');
        expect(result.mcpTools).toHaveLength(0);
        expect(result.estimatedTokens).toBe(1000);
      });
    });

    test('should classify complex requests correctly', () => {
      const complexRequests = [
        'Analyze the performance differences between React and Vue',
        'Compare different database architectures for scalability',
        'Evaluate the trade-offs of microservices vs monoliths',
        'Design a system with multiple steps to handle user authentication'
      ];

      complexRequests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('complex');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
        expect(result.requiredModules).toEqual(['SAGE', 'SEIQF', 'cognitive-tools/analysis']);
        expect(result.suggestedAgents).toContain('analyzer');
        expect(result.mcpTools).toContain('sequentialthinking');
        expect(result.estimatedTokens).toBe(3000);
      });
    });

    test('should classify search requests correctly', () => {
      const searchRequests = [
        'Find all JavaScript files in the project',
        'Search for the implementation of the login function',
        'Locate the configuration files',
        'Where is the database connection code?',
        'List all test files in the directory'
      ];

      searchRequests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('search');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
        expect(result.requiredModules).toEqual(['SIA', 'cognitive-tools/search']);
        expect(result.suggestedAgents).toContain('searcher');
        expect(result.mcpTools).toContain('tavily-search');
        expect(result.estimatedTokens).toBe(2000);
      });
    });

    test('should classify code requests correctly', () => {
      const codeRequests = [
        'Write a function to validate email addresses',
        'Implement a binary search algorithm',
        'Fix the bug in the authentication module',
        'Debug the memory leak in the application',
        'Refactor the user service class'
      ];

      codeRequests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('code');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
        expect(result.requiredModules).toEqual(['SEIQF', 'cognitive-tools/code-analysis']);
        expect(result.suggestedAgents).toContain('developer');
        expect(result.mcpTools).toContain('debuggingapproach');
        expect(result.estimatedTokens).toBe(2500);
      });
    });

    test('should classify meta requests correctly', () => {
      const metaRequests = [
        'How do you approach problem solving?',
        'What is your process for debugging?',
        'Why did you choose that implementation?',
        'Can you explain your reasoning?'
      ];

      metaRequests.forEach(request => {
        const result = classifyRequest(request);
        expect(result.category).toBe('meta');
        expect(result.confidence).toBeGreaterThanOrEqual(0.8);
        expect(result.requiredModules).toEqual(['SAGE', 'cognitive-tools/meta-reasoning']);
        expect(result.suggestedAgents).toContain('meta-reasoner');
        expect(result.mcpTools).toContain('metacognitivemonitoring');
        expect(result.estimatedTokens).toBe(1500);
      });
    });

    test('should handle ambiguous classifications with fallback', () => {
      const ambiguousRequest = 'tell me something';
      const result = classifyRequest(ambiguousRequest);
      
      expect(result.category).toBe('simple');
      expect(result.confidence).toBe(0.8);
      expect(result.requiredModules).toContain('response-formats');
    });

    test('should apply weighted scoring for file extensions in code requests', () => {
      const requestWithJS = 'Fix the error in app.js file';
      const requestWithPY = 'Debug the issue in main.py';
      
      const jsResult = classifyRequest(requestWithJS);
      const pyResult = classifyRequest(requestWithPY);
      
      expect(jsResult.category).toBe('code');
      expect(pyResult.category).toBe('code');
      expect(jsResult.confidence).toBeGreaterThanOrEqual(0.8);
      expect(pyResult.confidence).toBeGreaterThanOrEqual(0.8);
    });

    test('should handle edge cases gracefully', () => {
      const edgeCases = [
        '',
        '   ',
        '!!!',
        '12345',
        'a'.repeat(1000)
      ];

      edgeCases.forEach(request => {
        const result = classifyRequest(request);
        expect(result).toBeDefined();
        expect(result.category).toBeDefined();
        expect(result.confidence).toBeDefined();
        expect(result.requiredModules).toBeInstanceOf(Array);
      });
    });
  });

  describe('LRU Cache', () => {
    test('should cache and retrieve values correctly', () => {
      const cache = new LRUCache(3);
      
      cache.set('key1', 'value1');
      cache.set('key2', 'value2');
      cache.set('key3', 'value3');
      
      expect(cache.get('key1')).toBe('value1');
      expect(cache.get('key2')).toBe('value2');
      expect(cache.get('key3')).toBe('value3');
    });

    test('should evict least recently used items when capacity exceeded', () => {
      const cache = new LRUCache(2);
      
      cache.set('key1', 'value1');
      cache.set('key2', 'value2');
      cache.set('key3', 'value3'); // Should evict key1
      
      expect(cache.get('key1')).toBeNull();
      expect(cache.get('key2')).toBe('value2');
      expect(cache.get('key3')).toBe('value3');
    });

    test('should update LRU order on get', () => {
      const cache = new LRUCache(2);
      
      cache.set('key1', 'value1');
      cache.set('key2', 'value2');
      cache.get('key1'); // Makes key1 most recently used
      cache.set('key3', 'value3'); // Should evict key2
      
      expect(cache.get('key1')).toBe('value1');
      expect(cache.get('key2')).toBeNull();
      expect(cache.get('key3')).toBe('value3');
    });
  });

  describe('Module Mapping', () => {
    test('should return correct modules for each category', () => {
      expect(getRequiredModules('simple')).toEqual(['response-formats']);
      expect(getRequiredModules('complex')).toEqual(['SAGE', 'SEIQF', 'cognitive-tools/analysis']);
      expect(getRequiredModules('search')).toEqual(['SIA', 'cognitive-tools/search']);
      expect(getRequiredModules('code')).toEqual(['SEIQF', 'cognitive-tools/code-analysis']);
      expect(getRequiredModules('meta')).toEqual(['SAGE', 'cognitive-tools/meta-reasoning']);
    });

    test('should return correct priority for each category', () => {
      expect(getModulePriority('simple')).toBe(1.0);
      expect(getModulePriority('complex')).toBe(0.9);
      expect(getModulePriority('search')).toBe(0.95);
      expect(getModulePriority('code')).toBe(0.95);
      expect(getModulePriority('meta')).toBe(0.85);
    });

    test('should fallback to simple for unknown categories', () => {
      expect(getRequiredModules('unknown')).toEqual(['response-formats']);
      expect(getModulePriority('unknown')).toBe(1.0);
    });
  });

  describe('Performance', () => {
    test('should classify requests within 100ms', () => {
      const requests = [
        'What is JavaScript?',
        'Analyze the architecture of this system',
        'Find all test files',
        'Write a sorting algorithm',
        'How do you think?'
      ];

      requests.forEach(request => {
        const start = Date.now();
        classifyRequest(request);
        const elapsed = Date.now() - start;
        expect(elapsed).toBeLessThan(100);
      });
    });

    test('should benefit from caching on repeated requests', () => {
      const request = 'Write a function to parse JSON';
      
      // First call - no cache
      const start1 = Date.now();
      const result1 = classifyRequest(request);
      const time1 = Date.now() - start1;
      
      // Second call - should hit cache
      const start2 = Date.now();
      const result2 = classifyRequest(request);
      const time2 = Date.now() - start2;
      
      expect(result1).toEqual(result2);
      expect(time2).toBeLessThanOrEqual(time1);
    });
  });

  describe('Secure Cache Key Generation', () => {
    test('should generate consistent keys for same input', () => {
      const input = 'Test request';
      const key1 = generateSecureCacheKey(input);
      const key2 = generateSecureCacheKey(input);
      expect(key1).toBe(key2);
    });

    test('should generate different keys for different inputs', () => {
      const key1 = generateSecureCacheKey('Request 1');
      const key2 = generateSecureCacheKey('Request 2');
      expect(key1).not.toBe(key2);
    });

    test('should handle Unicode correctly', () => {
      const key1 = generateSecureCacheKey('café');
      const key2 = generateSecureCacheKey('cafe');
      expect(key1).not.toBe(key2);
    });

    test('should include cache prefix', () => {
      const key = generateSecureCacheKey('test');
      expect(key).toMatch(new RegExp(`^${CACHE_CONFIG.KEY_PREFIX}`));
    });
  });

  describe('TelemetryLogger', () => {
    test('should respect max entries limit', () => {
      const logger = new TelemetryLogger(5);
      
      for (let i = 0; i < 10; i++) {
        logger.log('test_event', { index: i });
      }
      
      const entries = logger.getEntries();
      expect(entries).toHaveLength(5);
      expect(entries[0].data.index).toBe(5); // Oldest should be index 5
      expect(entries[4].data.index).toBe(9); // Newest should be index 9
    });

    test('should create immutable entry copies', () => {
      const logger = new TelemetryLogger();
      logger.log('test', { value: 1 });
      
      const entries1 = logger.getEntries();
      const entries2 = logger.getEntries();
      
      expect(entries1).not.toBe(entries2); // Different array references
      expect(entries1).toEqual(entries2); // Same content
    });

    test('should clear entries', () => {
      const logger = new TelemetryLogger();
      logger.log('test1', {});
      logger.log('test2', {});
      
      expect(logger.getEntries()).toHaveLength(2);
      
      logger.clear();
      expect(logger.getEntries()).toHaveLength(0);
    });
  });

  describe('Constants', () => {
    test('should have valid confidence weights', () => {
      expect(CONFIDENCE_WEIGHTS.INDICATOR_MATCH).toBe(0.3);
      expect(CONFIDENCE_WEIGHTS.CODE_INDICATOR_MATCH).toBe(0.4);
      expect(CONFIDENCE_WEIGHTS.FILE_INDICATOR_MATCH).toBe(0.3);
      expect(CONFIDENCE_WEIGHTS.SELF_REFERENCE_MATCH).toBe(0.3);
      expect(CONFIDENCE_WEIGHTS.FALLBACK_THRESHOLD).toBe(0.8);
      expect(CONFIDENCE_WEIGHTS.FALLBACK_CONFIDENCE).toBe(0.8);
    });

    test('should have valid cache configuration', () => {
      expect(CACHE_CONFIG.MAX_SIZE).toBe(100);
      expect(CACHE_CONFIG.KEY_PREFIX).toBe('clf_');
    });

    test('should have valid telemetry configuration', () => {
      expect(TELEMETRY_CONFIG.MAX_ENTRIES).toBe(1000);
      expect(TELEMETRY_CONFIG.PERSIST_THRESHOLD).toBe(100);
    });
  });
});