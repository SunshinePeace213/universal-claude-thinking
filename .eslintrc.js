module.exports = {
  env: {
    node: true,
    jest: true,
    es2021: true
  },
  extends: ['eslint:recommended'],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  rules: {
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-unused-vars': ['error', { argsIgnorePattern: '^_' }]
  },
  overrides: [
    {
      files: ['**/*.bench.js', '**/performance/**/*.js'],
      rules: {
        'no-console': 'off'
      }
    }
  ],
  ignorePatterns: [
    '.claude/**/*.md',
    'docs/**/*.md',
    '*.md'
  ]
};