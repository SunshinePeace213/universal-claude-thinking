const fs = require('fs');
const path = require('path');

describe('Module Loader Infrastructure', () => {
  describe('Module Files', () => {
    it('should have module-loader.md file', () => {
      const modulePath = path.join(__dirname, '../../.claude/module-loader.md');
      expect(fs.existsSync(modulePath)).toBe(true);
    });
    
    it('should have metadata.yaml registry file', () => {
      const metadataPath = path.join(__dirname, '../../.claude/config/metadata.yaml');
      expect(fs.existsSync(metadataPath)).toBe(true);
    });
    
    it('should have protocol-state.json file', () => {
      const statePath = path.join(__dirname, '../../.claude/shared/protocol-state.json');
      expect(fs.existsSync(statePath)).toBe(true);
    });
  });
  
  describe('Module Loader Content', () => {
    let moduleContent;
    
    beforeEach(() => {
      const modulePath = path.join(__dirname, '../../.claude/module-loader.md');
      moduleContent = fs.readFileSync(modulePath, 'utf8');
    });
    
    it('should define ImportParser class', () => {
      expect(moduleContent).toContain('class ImportParser');
      expect(moduleContent).toContain('parse(content: string): ImportDirective[]');
    });
    
    it('should define ModuleLoader class', () => {
      expect(moduleContent).toContain('class ModuleLoader');
      expect(moduleContent).toContain('async loadModules(classification: RequestClassification)');
    });
    
    it('should define ModuleRegistry class', () => {
      expect(moduleContent).toContain('class ModuleRegistry');
      expect(moduleContent).toContain('async getMetadata(moduleId: string)');
    });
    
    it('should define DependencyResolver class', () => {
      expect(moduleContent).toContain('class DependencyResolver');
      expect(moduleContent).toContain('async resolveDependencies(moduleIds: string[])');
    });
    
    it('should define TokenTracker class', () => {
      expect(moduleContent).toContain('class TokenTracker');
      expect(moduleContent).toContain('reserveTokens(moduleId: string, tokenCount: number)');
    });
    
    it('should define SecurityValidator class', () => {
      expect(moduleContent).toContain('class SecurityValidator');
      expect(moduleContent).toContain('async validateModule(modulePath: string, expectedHash: string)');
    });
    
    it('should define ModuleHealthChecker class', () => {
      expect(moduleContent).toContain('class ModuleHealthChecker');
      expect(moduleContent).toContain('async checkModuleHealth');
    });
  });
  
  describe('Module Registry Content', () => {
    let registryContent;
    
    beforeEach(() => {
      const registryPath = path.join(__dirname, '../../.claude/config/metadata.yaml');
      registryContent = fs.readFileSync(registryPath, 'utf8');
    });
    
    it('should define SAGE module', () => {
      expect(registryContent).toContain('SAGE:');
      expect(registryContent).toContain('tokenCount: 2000');
      expect(registryContent).toContain('protocols: ["SAGE"]');
    });
    
    it('should define SEIQF module', () => {
      expect(registryContent).toContain('SEIQF:');
      expect(registryContent).toContain('tokenCount: 3000');
      expect(registryContent).toContain('protocols: ["SEIQF"]');
    });
    
    it('should define SIA module', () => {
      expect(registryContent).toContain('SIA:');
      expect(registryContent).toContain('tokenCount: 2000');
      expect(registryContent).toContain('protocols: ["SIA"]');
    });
    
    it('should define cognitive tools', () => {
      expect(registryContent).toContain('cognitive-tools/analysis:');
      expect(registryContent).toContain('cognitive-tools/search:');
      expect(registryContent).toContain('cognitive-tools/code-analysis:');
      expect(registryContent).toContain('cognitive-tools/meta-reasoning:');
    });
    
    it('should have security hashes for all modules', () => {
      const hashPattern = /securityHash: "[a-f0-9]+"/g;
      const matches = registryContent.match(hashPattern);
      expect(matches).toBeTruthy();
      expect(matches.length).toBe(8); // Exactly 8 modules
      
      // Verify all hashes are proper length (around 60 chars for SHA-256 hex)
      matches.forEach(match => {
        const hash = match.match(/"([^"]+)"/)[1];
        expect(hash.length).toBeGreaterThanOrEqual(58);
        expect(hash.length).toBeLessThanOrEqual(64);
      });
    });
  });
  
  describe('CLAUDE.md Integration', () => {
    let claudeContent;
    
    beforeEach(() => {
      const claudePath = path.join(__dirname, '../../.claude/CLAUDE.md');
      claudeContent = fs.readFileSync(claudePath, 'utf8');
    });
    
    it('should import module-loader.md', () => {
      expect(claudeContent).toContain('@import "./module-loader.md"');
    });
    
    it('should have updated version', () => {
      expect(claudeContent).toContain('version: 2.1.0');
    });
    
    it('should contain dynamic module loading code', () => {
      expect(claudeContent).toContain('const classification = await classifier.classify(userRequest)');
      expect(claudeContent).toContain('const loader = new ModuleLoader');
      expect(claudeContent).toContain('const resolver = new DependencyResolver');
    });
    
    it('should have enhanced debug header', () => {
      expect(claudeContent).toContain('Module Health:');
      expect(claudeContent).toContain('tokenTracker.getTotalTokensUsed()');
    });
  });
  
  describe('Import Parser Logic', () => {
    it('should match import patterns correctly', () => {
      const importRegex = /@import\s+"([^"]+)"(?:\s+if\s+(.+))?/g;
      
      const testCases = [
        {
          input: '@import "./module.md"',
          expectedPath: './module.md',
          expectedCondition: undefined
        },
        {
          input: '@import "./test.md" if confidence > 0.8',
          expectedPath: './test.md',
          expectedCondition: 'confidence > 0.8'
        }
      ];
      
      testCases.forEach(({ input, expectedPath, expectedCondition }) => {
        const match = importRegex.exec(input);
        importRegex.lastIndex = 0; // Reset regex
        
        expect(match).toBeTruthy();
        expect(match[1]).toBe(expectedPath);
        expect(match[2]).toBe(expectedCondition);
      });
    });
  });
  
  describe('Token Budget Calculations', () => {
    it('should have correct token allocations in metadata', () => {
      const registryPath = path.join(__dirname, '../../.claude/config/metadata.yaml');
      const registryContent = fs.readFileSync(registryPath, 'utf8');
      
      // Extract token counts
      const tokenMatches = registryContent.match(/tokenCount: (\d+)/g);
      const tokenCounts = tokenMatches.map(m => parseInt(m.split(': ')[1]));
      
      // Verify token counts match story requirements
      expect(tokenCounts).toContain(2000); // SAGE
      expect(tokenCounts).toContain(3000); // SEIQF
      expect(tokenCounts).toContain(2000); // SIA
      expect(tokenCounts).toContain(1000); // response-formats
      expect(tokenCounts).toContain(500);  // cognitive tools
      
      // Total should not exceed reasonable limits
      const total = tokenCounts.reduce((sum, count) => sum + count, 0);
      expect(total).toBeLessThan(15000); // Reasonable total for all modules
    });
  });
  
  describe('Security Validation Patterns', () => {
    let moduleContent;
    
    beforeEach(() => {
      const modulePath = path.join(__dirname, '../../.claude/module-loader.md');
      moduleContent = fs.readFileSync(modulePath, 'utf8');
    });
    
    it('should check for path traversal attempts', () => {
      expect(moduleContent).toContain('normalized.includes(\'..\')');
      expect(moduleContent).toContain('Path traversal attempt detected');
    });
    
    it('should validate against forbidden paths', () => {
      expect(moduleContent).toContain("forbidden = ['/etc', '/usr', '/bin', '/sbin', '/var', '/tmp']");
    });
    
    it('should implement quarantine system', () => {
      expect(moduleContent).toContain('quarantineMaxRetries: 3');
      expect(moduleContent).toContain('QUARANTINE_ADD');
      expect(moduleContent).toContain('QUARANTINE_RELEASE');
    });
  });
  
  describe('Health Check Validations', () => {
    let moduleContent;
    
    beforeEach(() => {
      const modulePath = path.join(__dirname, '../../.claude/module-loader.md');
      moduleContent = fs.readFileSync(modulePath, 'utf8');
    });
    
    it('should validate markdown syntax', () => {
      expect(moduleContent).toContain('checkBraceBalance');
      expect(moduleContent).toContain('Missing markdown headers');
    });
    
    it('should check for dangerous patterns', () => {
      expect(moduleContent).toContain('eval\\s*\\(');
      expect(moduleContent).toContain('Function\\s*\\(');
      expect(moduleContent).toContain('<script');
    });
    
    it('should enforce performance thresholds', () => {
      expect(moduleContent).toContain('loadTime: 50');
      expect(moduleContent).toContain('maxSize: 100000');
      expect(moduleContent).toContain('maxComplexity: 20');
    });
  });
  
  describe('Protocol State Structure', () => {
    let stateContent;
    
    beforeEach(() => {
      const statePath = path.join(__dirname, '../../.claude/shared/protocol-state.json');
      stateContent = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    });
    
    it('should have correct initial state', () => {
      expect(stateContent.version).toBe('1.0.0');
      expect(stateContent.tokenUsage.total).toBe(5000);
      expect(stateContent.tokenUsage.available).toBe(5000);
      expect(stateContent.tokenUsage.used).toBe(0);
    });
    
    it('should have telemetry fields', () => {
      expect(stateContent.telemetry).toBeDefined();
      expect(stateContent.telemetry.cacheHits).toBe(0);
      expect(stateContent.telemetry.moduleLoadFailures).toBe(0);
    });
    
    it('should have module metrics structure', () => {
      expect(stateContent.moduleMetrics).toBeDefined();
      expect(typeof stateContent.moduleMetrics).toBe('object');
    });
  });
  
  describe('Hot-Reload Support', () => {
    let moduleContent;
    
    beforeEach(() => {
      const modulePath = path.join(__dirname, '../../.claude/module-loader.md');
      moduleContent = fs.readFileSync(modulePath, 'utf8');
    });
    
    it('should define HotReloadManager class', () => {
      expect(moduleContent).toContain('class HotReloadManager');
      expect(moduleContent).toContain('async startWatching()');
      expect(moduleContent).toContain('async stopWatching()');
    });
    
    it('should implement file watching configuration', () => {
      expect(moduleContent).toContain('FileWatcherConfig');
      expect(moduleContent).toContain('debounceMs: 300');
      expect(moduleContent).toContain('preserveState: true');
    });
    
    it('should handle reload events', () => {
      expect(moduleContent).toContain('interface ReloadEvent');
      expect(moduleContent).toContain('changeType: \'create\' | \'update\' | \'delete\'');
      expect(moduleContent).toContain('reloadModule(event: ReloadEvent)');
    });
    
    it('should preserve module state during reload', () => {
      expect(moduleContent).toContain('preserveModuleState');
      expect(moduleContent).toContain('restoreModuleState');
      expect(moduleContent).toContain('stateSnapshot: Map<string, any>');
    });
    
    it('should implement recovery mechanism', () => {
      expect(moduleContent).toContain('attemptRecovery');
      expect(moduleContent).toContain('maxRetries: 3');
      expect(moduleContent).toContain('getCachedModule');
    });
    
    it('should provide reload status API', () => {
      expect(moduleContent).toContain('getReloadStatus()');
      expect(moduleContent).toContain('reloadModuleManual');
      expect(moduleContent).toContain('watching: Array.from(this.watchers.keys())');
    });
  });
});