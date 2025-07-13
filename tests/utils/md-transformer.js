// Custom Jest transformer for .md files containing JavaScript code

module.exports = {
  process(sourceText) {
    // Extract JavaScript code blocks from markdown
    const jsCodeBlocks = [];
    const jsRegex = /```javascript\n([\s\S]*?)```/g;
    let match;
    
    while ((match = jsRegex.exec(sourceText)) !== null) {
      jsCodeBlocks.push(match[1]);
    }
    
    // Combine all JavaScript code blocks
    const combinedCode = jsCodeBlocks.join('\n\n');
    
    // Return the JavaScript code for Jest to execute
    return {
      code: combinedCode
    };
  }
};