// Jest setup file for global test configuration

// Increase timeout for async operations
jest.setTimeout(10000);

// Mock file system operations for security
const fs = require('fs');
const path = require('path');

// Store original methods
const originalReadFileSync = fs.readFileSync;
const originalExistsSync = fs.existsSync;

// Create mock file system state
const mockFileSystem = new Map();

// Helper to add mock files
global.addMockFile = (filePath, content) => {
  mockFileSystem.set(path.resolve(filePath), content);
};

// Helper to clear mock files
global.clearMockFiles = () => {
  mockFileSystem.clear();
};

// Mock fs.readFileSync to use mock files when available
fs.readFileSync = jest.fn((filePath, encoding) => {
  const resolvedPath = path.resolve(filePath);
  if (mockFileSystem.has(resolvedPath)) {
    return mockFileSystem.get(resolvedPath);
  }
  return originalReadFileSync(filePath, encoding);
});

// Mock fs.existsSync to check mock files
fs.existsSync = jest.fn(filePath => {
  const resolvedPath = path.resolve(filePath);
  if (mockFileSystem.has(resolvedPath)) {
    return true;
  }
  return originalExistsSync(filePath);
});

// Global test helpers
global.expectTokenCount = (content, maxTokens) => {
  const tokenCount = content.split(/\s+/).length;
  expect(tokenCount).toBeLessThanOrEqual(maxTokens);
};

global.expectValidPath = filePath => {
  expect(filePath).toMatch(/^\.claude\//);
  expect(filePath).not.toMatch(/\.\./);
  expect(filePath).not.toMatch(/^\/|^\\/);
};

// Console override for cleaner test output
const originalConsole = global.console;
global.console = {
  ...originalConsole,
  log: jest.fn(),
  warn: jest.fn(),
  error: jest.fn()
};

// Cleanup after each test
afterEach(() => {
  jest.clearAllMocks();
  global.clearMockFiles();
});
