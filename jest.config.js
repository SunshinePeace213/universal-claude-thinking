module.exports = {
  testEnvironment: 'node',
  testMatch: [
    '**/tests/**/*.test.js',
    '**/tests/**/*.bench.js'
  ],
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    '.claude/**/*.md',
    '!.claude/thinking-modules/**',
    '!.claude/cognitive-tools/**'
  ],
  transform: {
    '\\.md$': '<rootDir>/tests/utils/md-transformer.js'
  },
  moduleFileExtensions: ['js', 'json', 'md'],
  testTimeout: 10000,
  verbose: true
};