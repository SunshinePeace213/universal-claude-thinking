const { classifyRequest } = require('../../.claude/request-classifier.md');

const ITERATIONS = 1000;
const PERFORMANCE_THRESHOLD = 100; // 100ms per classification

describe('Classification Performance Benchmarks', () => {
  const testRequests = {
    simple: [
      'What is Node.js?',
      'How do I create a variable?',
      'Explain async/await',
      'Define microservices'
    ],
    complex: [
      'Analyze the performance implications of using Redux vs Context API',
      'Compare different caching strategies for distributed systems',
      'Evaluate the architectural patterns for building scalable APIs',
      'Design a multi-step process for data migration'
    ],
    search: [
      'Find all TypeScript files in the src directory',
      'Search for implementations of the Observer pattern',
      'Locate the database configuration files',
      'Where are the authentication middleware functions?'
    ],
    code: [
      'Write a recursive function to calculate fibonacci numbers',
      'Implement a rate limiting middleware for Express.js',
      'Fix the memory leak in the WebSocket handler',
      'Refactor the user authentication service'
    ],
    meta: [
      'How do you approach debugging complex issues?',
      'What is your process for code review?',
      'Why did you suggest that architecture?',
      'Can you explain your thinking process?'
    ]
  };

  function benchmarkCategory(category, requests) {
    const times = [];
    
    for (let i = 0; i < ITERATIONS; i++) {
      const request = requests[i % requests.length];
      const start = process.hrtime.bigint();
      classifyRequest(request);
      const end = process.hrtime.bigint();
      times.push(Number(end - start) / 1000000); // Convert to milliseconds
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const maxTime = Math.max(...times);
    const minTime = Math.min(...times);
    const p95Time = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];
    
    return { avgTime, maxTime, minTime, p95Time };
  }

  test('should meet performance requirements for all categories', () => {
    const results = {};
    
    Object.entries(testRequests).forEach(([category, requests]) => {
      results[category] = benchmarkCategory(category, requests);
    });
    
    console.log('\nPerformance Benchmark Results:');
    console.log('==============================');
    
    Object.entries(results).forEach(([category, metrics]) => {
      console.log(`\n${category.toUpperCase()} requests:`);
      console.log(`  Average: ${metrics.avgTime.toFixed(2)}ms`);
      console.log(`  Min: ${metrics.minTime.toFixed(2)}ms`);
      console.log(`  Max: ${metrics.maxTime.toFixed(2)}ms`);
      console.log(`  P95: ${metrics.p95Time.toFixed(2)}ms`);
      
      expect(metrics.avgTime).toBeLessThan(PERFORMANCE_THRESHOLD);
      expect(metrics.p95Time).toBeLessThan(PERFORMANCE_THRESHOLD);
    });
  });

  test('should demonstrate cache effectiveness', () => {
    const request = 'Write a function to validate email addresses';
    const times = { cold: [], warm: [] };
    
    // Cold cache runs
    for (let i = 0; i < 100; i++) {
      // Create unique request to avoid cache
      const uniqueRequest = `${request} ${i}`;
      const start = process.hrtime.bigint();
      classifyRequest(uniqueRequest);
      const end = process.hrtime.bigint();
      times.cold.push(Number(end - start) / 1000000);
    }
    
    // Warm cache runs
    for (let i = 0; i < 100; i++) {
      const start = process.hrtime.bigint();
      classifyRequest(request); // Same request, should hit cache
      const end = process.hrtime.bigint();
      times.warm.push(Number(end - start) / 1000000);
    }
    
    const avgCold = times.cold.reduce((a, b) => a + b, 0) / times.cold.length;
    const avgWarm = times.warm.reduce((a, b) => a + b, 0) / times.warm.length;
    
    console.log('\nCache Performance:');
    console.log('==================');
    console.log(`Cold cache average: ${avgCold.toFixed(2)}ms`);
    console.log(`Warm cache average: ${avgWarm.toFixed(2)}ms`);
    console.log(`Performance improvement: ${((1 - avgWarm/avgCold) * 100).toFixed(1)}%`);
    
    expect(avgWarm).toBeLessThan(avgCold);
    expect(avgWarm).toBeLessThan(10); // Cached lookups should be very fast
  });

  test('should handle mixed workload efficiently', () => {
    const allRequests = Object.values(testRequests).flat();
    const times = [];
    
    for (let i = 0; i < ITERATIONS; i++) {
      const request = allRequests[Math.floor(Math.random() * allRequests.length)];
      const start = process.hrtime.bigint();
      classifyRequest(request);
      const end = process.hrtime.bigint();
      times.push(Number(end - start) / 1000000);
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const p99Time = times.sort((a, b) => a - b)[Math.floor(times.length * 0.99)];
    
    console.log('\nMixed Workload Performance:');
    console.log('===========================');
    console.log(`Average: ${avgTime.toFixed(2)}ms`);
    console.log(`P99: ${p99Time.toFixed(2)}ms`);
    
    expect(avgTime).toBeLessThan(PERFORMANCE_THRESHOLD);
    expect(p99Time).toBeLessThan(PERFORMANCE_THRESHOLD * 1.5); // Allow some headroom for P99
  });

  test('should maintain performance with long requests', () => {
    const longRequests = [
      'a'.repeat(500) + ' write a function',
      'implement ' + 'complex '.repeat(50) + 'algorithm',
      'find all ' + 'nested '.repeat(30) + 'files'
    ];
    
    const times = [];
    
    longRequests.forEach(request => {
      const start = process.hrtime.bigint();
      classifyRequest(request);
      const end = process.hrtime.bigint();
      times.push(Number(end - start) / 1000000);
    });
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    
    console.log('\nLong Request Performance:');
    console.log('========================');
    console.log(`Average: ${avgTime.toFixed(2)}ms`);
    
    expect(avgTime).toBeLessThan(PERFORMANCE_THRESHOLD);
  });
});