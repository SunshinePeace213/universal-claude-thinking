// Jest transformer for markdown files
// This allows Jest to process .md files in tests

const crypto = require('crypto');

module.exports = {
  process(sourceText) {
    // Simple transformation: export the markdown content as a string
    const content = JSON.stringify(sourceText);

    return {
      code: `module.exports = ${content};`
    };
  },

  getCacheKey(sourceText, sourcePath, configString) {
    // Create a cache key based on file content and config
    return crypto
      .createHash('md5')
      .update(sourceText)
      .update(configString)
      .digest('hex');
  }
};
