---
title: Testing and Quality Assurance
sidebar_position: 4
description: Comprehensive testing strategies for ensuring the quality and reliability of the textbook
duration: 180
difficulty: advanced
learning_objectives:
  - Implement comprehensive testing strategies for the textbook
  - Create unit, integration, and end-to-end tests
  - Establish quality assurance processes
  - Set up automated testing and CI/CD pipelines
  - Monitor and maintain quality standards
---

# Testing and Quality Assurance

## Learning Objectives

By the end of this guide, you will be able to:
- Implement comprehensive testing strategies for educational content
- Create effective unit, integration, and end-to-end tests
- Establish quality assurance processes for the textbook
- Set up automated testing and CI/CD pipelines
- Monitor and maintain quality standards for ongoing development

## Testing Philosophy and Strategy

### Importance of Testing in Educational Content

Testing is crucial for educational content because:

1. **Accuracy**: Educational content must be technically accurate
2. **Consistency**: Content should follow consistent patterns and standards
3. **Accessibility**: Content must be accessible to all learners
4. **Reliability**: The platform must function consistently
5. **Performance**: Content should load and function efficiently
6. **Compatibility**: Content must work across different devices and browsers

### Testing Pyramid for Educational Platforms

```
    End-to-End Tests (5-10%)
         ↓
    Integration Tests (20-30%)
         ↓
    Unit Tests (60-70%)
```

The testing pyramid for our textbook platform follows this structure:

- **Unit Tests**: Test individual components and functions (60-70% of tests)
- **Integration Tests**: Test how components work together (20-30% of tests)
- **End-to-End Tests**: Test complete user workflows (5-10% of tests)

## Unit Testing

### Component Testing

Test individual React components for proper functionality:

```tsx
// tests/components/InteractiveDemo.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import InteractiveDemo from '../../src/components/InteractiveDemo/InteractiveDemo';

describe('InteractiveDemo Component', () => {
  const defaultProps = {
    defaultCode: 'print("Hello, World!")',
    language: 'python',
    title: 'Test Demo'
  };

  it('renders with default props', () => {
    render(<InteractiveDemo {...defaultProps} />);

    expect(screen.getByText('Test Demo')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('print("Hello, World!")');
  });

  it('updates code when textarea changes', () => {
    render(<InteractiveDemo {...defaultProps} />);

    const textbox = screen.getByRole('textbox');
    fireEvent.change(textbox, { target: { value: 'console.log("Updated");' } });

    expect(textbox).toHaveValue('console.log("Updated");');
  });

  it('executes code when run button is clicked', async () => {
    render(<InteractiveDemo {...defaultProps} />);

    const runButton = screen.getByText('Run Code');
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(screen.getByText(/output:/i)).toBeInTheDocument();
    });
  });

  it('resets code when reset button is clicked', () => {
    render(<InteractiveDemo {...defaultProps} />);

    const textbox = screen.getByRole('textbox');
    fireEvent.change(textbox, { target: { value: 'modified code' } });

    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);

    expect(textbox).toHaveValue('print("Hello, World!")');
  });
});
```

### Utility Function Testing

Test utility functions that support the textbook:

```tsx
// tests/utils/contentUtils.test.ts
import { validateCode, extractLearningObjectives, calculateReadingTime } from '../../src/utils/contentUtils';

describe('Content Utilities', () => {
  describe('validateCode', () => {
    it('validates Python code correctly', () => {
      const validCode = 'def hello():\n    return "Hello"';
      const invalidCode = 'def hello(:\n    return "Hello"'; // Missing closing parenthesis

      expect(validateCode(validCode, 'python')).toBe(true);
      expect(validateCode(invalidCode, 'python')).toBe(false);
    });

    it('validates JavaScript code correctly', () => {
      const validCode = 'function hello() {\n    return "Hello";\n}';
      const invalidCode = 'function hello() {\n    return "Hello";'; // Missing closing brace

      expect(validateCode(validCode, 'javascript')).toBe(true);
      expect(validateCode(invalidCode, 'javascript')).toBe(false);
    });
  });

  describe('extractLearningObjectives', () => {
    it('extracts learning objectives from markdown', () => {
      const markdown = `
# Lesson Title
## Learning Objectives
- Understand basic concepts
- Learn how to implement solutions
- Apply knowledge practically
      `;

      const objectives = extractLearningObjectives(markdown);
      expect(objectives).toEqual([
        'Understand basic concepts',
        'Learn how to implement solutions',
        'Apply knowledge practically'
      ]);
    });

    it('handles empty learning objectives', () => {
      const markdown = '# Lesson Title\nNo objectives here';
      const objectives = extractLearningObjectives(markdown);
      expect(objectives).toEqual([]);
    });
  });

  describe('calculateReadingTime', () => {
    it('calculates reading time for English text', () => {
      const text = 'This is a sample text that would take approximately one minute to read.';
      const time = calculateReadingTime(text, 'en');
      expect(time).toBeGreaterThan(0);
      expect(time).toBeLessThanOrEqual(2); // Should be around 1 minute
    });

    it('calculates reading time for longer text', () => {
      const longText = 'This is a sample text. '.repeat(300); // About 600 words
      const time = calculateReadingTime(longText, 'en');
      expect(time).toBeGreaterThanOrEqual(2); // Should be at least 2 minutes
    });
  });
});
```

### Content Validation Testing

Test content validation and processing:

```tsx
// tests/utils/contentValidation.test.ts
import {
  validateMarkdownContent,
  checkAccessibility,
  validateCodeSnippets,
  verifyExternalLinks
} from '../../src/utils/contentValidation';

describe('Content Validation', () => {
  describe('validateMarkdownContent', () => {
    it('validates proper markdown structure', () => {
      const validMarkdown = `
---
title: Test Lesson
sidebar_position: 1
---

# Test Lesson

## Section 1

Content here.
      `;

      const result = validateMarkdownContent(validMarkdown);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('detects missing required frontmatter', () => {
      const invalidMarkdown = `
# Test Lesson

Content here.
      `;

      const result = validateMarkdownContent(invalidMarkdown);
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Missing required frontmatter: title');
    });

    it('detects invalid markdown syntax', () => {
      const invalidMarkdown = `
---
title: Test Lesson
sidebar_position: 1
---

# Test Lesson

This has [broken link](without-closing-parenthesis
      `;

      const result = validateMarkdownContent(invalidMarkdown);
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Invalid markdown syntax detected');
    });
  });

  describe('checkAccessibility', () => {
    it('identifies accessibility issues', () => {
      const content = `
# Lesson

![Image without alt text]()

This content has poor contrast: [link](#) with #333 text on #444 background.
      `;

      const issues = checkAccessibility(content);
      expect(issues).toContain('Missing alt text for image');
      expect(issues).toContain('Insufficient color contrast');
    });

    it('passes content with good accessibility', () => {
      const content = `
# Lesson

![Descriptive alt text](image.jpg)

This content has good contrast.
      `;

      const issues = checkAccessibility(content);
      expect(issues).toHaveLength(0);
    });
  });

  describe('validateCodeSnippets', () => {
    it('validates Python code snippets', () => {
      const content = `
\`\`\`python
def hello():
    return "Hello, World!"
\`\`\`
      `;

      const result = validateCodeSnippets(content);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('detects syntax errors in code snippets', () => {
      const content = `
\`\`\`python
def hello(:
    return "Hello, World!"
\`\`\`
      `;

      const result = validateCodeSnippets(content);
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Syntax error in Python code snippet');
    });
  });

  describe('verifyExternalLinks', () => {
    it('checks external links', async () => {
      // Mock fetch responses
      global.fetch = jest.fn(() =>
        Promise.resolve({
          status: 200,
          ok: true,
        } as Response)
      ) as jest.Mock;

      const content = `
[Valid Link](https://example.com)
[Another Link](https://google.com)
      `;

      const brokenLinks = await verifyExternalLinks(content);
      expect(brokenLinks).toHaveLength(0);
    });
  });
});
```

## Integration Testing

### Component Integration Testing

Test how components work together:

```tsx
// tests/integration/LessonPage.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import LessonPage from '../../src/pages/LessonPage';
import { MemoryRouter } from 'react-router-dom';

describe('Lesson Page Integration', () => {
  const lessonData = {
    title: 'Introduction to ROS 2',
    content: `
---
title: Introduction to ROS 2
sidebar_position: 1
---

# Introduction to ROS 2

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the basics of ROS 2
- Create simple nodes
- Use topics and services

## Main Content

ROS 2 is a flexible framework for writing robot applications.
    `,
    duration: 45,
    difficulty: 'beginner',
    learningObjectives: [
      'Understand the basics of ROS 2',
      'Create simple nodes',
      'Use topics and services'
    ]
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('integrates all components properly', async () => {
    render(
      <MemoryRouter>
        <LessonPage lessonData={lessonData} />
      </MemoryRouter>
    );

    // Check that all components are rendered
    expect(screen.getByText('Introduction to ROS 2')).toBeInTheDocument();
    expect(screen.getByText('Learning Objectives')).toBeInTheDocument();
    expect(screen.getByText('Main Content')).toBeInTheDocument();

    // Check that interactive components are present
    expect(screen.queryByRole('button', { name: /run code/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /submit answer/i })).toBeInTheDocument();
  });

  it('displays learning objectives correctly', async () => {
    render(
      <MemoryRouter>
        <LessonPage lessonData={lessonData} />
      </MemoryRouter>
    );

    await waitFor(() => {
      lessonData.learningObjectives.forEach(objective => {
        expect(screen.getByText(objective)).toBeInTheDocument();
      });
    });
  });

  it('handles content with code snippets', () => {
    const lessonWithCode = {
      ...lessonData,
      content: `
---
title: Code Example
---

# Code Example

\`\`\`python
def hello_ros():
    print("Hello, ROS 2!")
\`\`\`
      `
    };

    render(
      <MemoryRouter>
        <LessonPage lessonData={lessonWithCode} />
      </MemoryRouter>
    );

    expect(screen.getByText(/def hello_ros/i)).toBeInTheDocument();
    expect(screen.getByText(/print\("Hello, ROS 2!"\)/i)).toBeInTheDocument();
  });
});
```

### API Integration Testing

Test integration with external services:

```tsx
// tests/integration/api.test.ts
import {
  fetchLessonContent,
  submitAssessment,
  trackUserProgress,
  searchContent
} from '../../src/api/textbookApi';

// Mock the fetch function
global.fetch = jest.fn();

describe('API Integration', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
  });

  describe('fetchLessonContent', () => {
    it('fetches lesson content successfully', async () => {
      const mockResponse = {
        title: 'Test Lesson',
        content: '# Test Content',
        metadata: { duration: 30, difficulty: 'intermediate' }
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await fetchLessonContent('test-lesson');

      expect(global.fetch).toHaveBeenCalledWith('/api/lessons/test-lesson', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      expect(result).toEqual(mockResponse);
    });

    it('handles fetch errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404
      });

      await expect(fetchLessonContent('nonexistent-lesson')).rejects.toThrow('Lesson not found');
    });
  });

  describe('submitAssessment', () => {
    it('submits assessment successfully', async () => {
      const mockSubmission = {
        userId: 'user123',
        lessonId: 'lesson456',
        answers: [0, 1, 2],
        score: 85
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, score: 85 })
      });

      const result = await submitAssessment(mockSubmission);

      expect(global.fetch).toHaveBeenCalledWith('/api/assessments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockSubmission)
      });
      expect(result).toEqual({ success: true, score: 85 });
    });
  });

  describe('trackUserProgress', () => {
    it('tracks user progress', async () => {
      const progressData = {
        userId: 'user123',
        lessonId: 'lesson456',
        completed: true,
        timeSpent: 1200 // seconds
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });

      const result = await trackUserProgress(progressData);

      expect(global.fetch).toHaveBeenCalledWith('/api/progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(progressData)
      });
      expect(result).toEqual({ success: true });
    });
  });

  describe('searchContent', () => {
    it('searches content successfully', async () => {
      const mockResults = [
        { id: 'lesson1', title: 'Introduction', content: 'Intro content' },
        { id: 'lesson2', title: 'Advanced Topics', content: 'Advanced content' }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ results: mockResults, count: 2 })
      });

      const result = await searchContent('introduction');

      expect(global.fetch).toHaveBeenCalledWith('/api/search?q=introduction', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      expect(result).toEqual({ results: mockResults, count: 2 });
    });
  });
});
```

## End-to-End Testing

### Puppeteer E2E Tests

Set up end-to-end tests using Puppeteer:

```typescript
// tests/e2e/homepage.test.ts
import puppeteer, { Browser, Page } from 'puppeteer';
import { execSync } from 'child_process';

describe('Homepage E2E Tests', () => {
  let browser: Browser;
  let page: Page;

  beforeAll(async () => {
    // Start the development server
    execSync('cd docs && npm start > /dev/null 2>&1 &', { cwd: process.cwd() });

    // Give the server time to start
    await new Promise(resolve => setTimeout(resolve, 5000));

    browser = await puppeteer.launch({
      headless: true, // Set to false for debugging
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
  });

  beforeEach(async () => {
    page = await browser.newPage();
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });
  });

  afterEach(async () => {
    await page.close();
  });

  afterAll(async () => {
    await browser.close();
    // Stop the development server if needed
  });

  it('loads homepage successfully', async () => {
    const title = await page.title();
    expect(title).toMatch(/Physical AI & Humanoid Robotics/);

    // Check for key elements
    const heroText = await page.$eval('h1', el => el.textContent);
    expect(heroText).toContain('Physical AI');

    const navigation = await page.$$('.navbar__item, .menu__link');
    expect(navigation.length).toBeGreaterThan(0);
  });

  it('navigates to a lesson page', async () => {
    // Click on a navigation link
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2' }),
      page.click('text="Week 1-2: Introduction"')
    ]);

    // Verify we're on a lesson page
    const url = page.url();
    expect(url).toMatch(/week-01-02/);

    const heading = await page.$eval('h1', el => el.textContent);
    expect(heading).toBeTruthy();
  });

  it('interacts with assessment component', async () => {
    await page.goto('http://localhost:3000/docs/week-01-02-introduction/foundations-of-physical-ai', {
      waitUntil: 'networkidle2'
    });

    // Check for assessment components
    const assessments = await page.$$('.assessment-container');
    if (assessments.length > 0) {
      // Click on an option
      await page.click('.option:first-child');

      // Submit the assessment
      await page.click('button[type="submit"]');

      // Check for feedback
      await page.waitForSelector('.result', { timeout: 5000 });
      const result = await page.$eval('.result', el => el.textContent);
      expect(result).toMatch(/correct|incorrect/i);
    }
  });

  it('searches for content', async () => {
    // Click search button
    await page.click('svg[aria-label="Search"]');

    // Wait for search modal
    await page.waitForSelector('input[placeholder="Search..."]', { timeout: 5000 });

    // Type search query
    await page.type('input[placeholder="Search..."]', 'ROS 2');

    // Wait for results
    await page.waitForSelector('.search-result-item', { timeout: 10000 });

    const results = await page.$$('.search-result-item');
    expect(results.length).toBeGreaterThan(0);
  });
});
```

### Cypress E2E Tests

Set up additional E2E tests with Cypress:

```typescript
// tests/e2e/lesson-interactions.cy.ts
describe('Lesson Interaction Tests', () => {
  beforeEach(() => {
    cy.visit('/docs/week-01-02-introduction/foundations-of-physical-ai');
  });

  it('should render the lesson content correctly', () => {
    cy.get('h1').should('be.visible');
    cy.get('[data-testid="lesson-content"]').should('exist');

    // Check for learning objectives
    cy.contains('Learning Objectives').should('be.visible');
    cy.get('[data-testid="learning-objective"]').should('have.length.greaterThan', 0);
  });

  it('should allow code execution in interactive demos', () => {
    cy.get('[data-testid="interactive-demo"]').first().within(() => {
      // Check that code editor is present
      cy.get('textarea').should('be.visible');

      // Check that run button exists
      cy.get('button').contains('Run Code').should('be.visible');

      // Optionally, test code execution
      cy.get('textarea').invoke('val').then((val) => {
        if (val && val.toString().includes('print')) {
          cy.get('button').contains('Run Code').click();
          cy.get('[data-testid="output-panel"]').should('be.visible');
        }
      });
    });
  });

  it('should handle assessment submissions', () => {
    cy.get('[data-testid="assessment-question"]').first().within(() => {
      // Select an option
      cy.get('[data-testid="assessment-option"]').first().click();

      // Submit the assessment
      cy.get('button').contains('Submit Answer').click();

      // Check for feedback
      cy.get('[data-testid="assessment-feedback"]').should('be.visible');
    });
  });

  it('should track lesson progress', () => {
    // Scroll to bottom to trigger progress tracking
    cy.get('main').scrollTo('bottom');

    // Check that progress tracking occurred
    // This would depend on your specific implementation
    cy.window().then((win) => {
      // Check if progress tracking function was called
      expect(win).to.have.property('trackProgress');
    });
  });
});

describe('Navigation Tests', () => {
  it('should navigate between lessons', () => {
    cy.visit('/docs/week-01-02-introduction/foundations-of-physical-ai');

    // Click next lesson link (assuming it exists)
    cy.contains('Next Lesson').click();
    cy.url().should('include', 'from-digital-ai-to-physical-systems');

    // Click previous lesson link
    cy.contains('Previous Lesson').click();
    cy.url().should('include', 'foundations-of-physical-ai');
  });

  it('should use sidebar navigation', () => {
    cy.visit('/');

    // Click on a sidebar item
    cy.get('.sidebar-item').first().click();

    // Verify navigation
    cy.url().should('include', 'docs');
    cy.get('h1').should('be.visible');
  });
});
```

## Accessibility Testing

### Automated Accessibility Tests

Set up automated accessibility testing:

```tsx
// tests/accessibility/automated.test.ts
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import '@testing-library/jest-dom';

expect.extend(toHaveNoViolations);

import HomePage from '../../src/pages/HomePage';
import LessonPage from '../../src/pages/LessonPage';
import InteractiveDemo from '../../src/components/InteractiveDemo/InteractiveDemo';
import Assessment from '../../src/components/Assessment/Assessment';

describe('Accessibility Tests', () => {
  describe('HomePage', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(<HomePage />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('LessonPage', () => {
    it('has no accessibility violations', async () => {
      const lessonData = {
        title: 'Test Lesson',
        content: '# Test Content',
        duration: 30,
        difficulty: 'beginner',
        learningObjectives: ['Objective 1', 'Objective 2']
      };

      const { container } = render(<LessonPage lessonData={lessonData} />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Interactive Components', () => {
    it('InteractiveDemo has no accessibility violations', async () => {
      const { container } = render(
        <InteractiveDemo
          defaultCode="console.log('test');"
          language="javascript"
          title="Test Demo"
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('Assessment has no accessibility violations', async () => {
      const { container } = render(
        <Assessment
          question="Test question?"
          type="single-choice"
          options={['Option 1', 'Option 2', 'Option 3']}
          correctIndex={0}
          explanation="This is the explanation."
        />
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
```

### Manual Accessibility Testing

Document manual accessibility testing procedures:

```markdown
# Manual Accessibility Testing Procedures

## Keyboard Navigation Testing

### Test Steps:
1. Navigate through the entire page using only the Tab key
2. Verify all interactive elements receive focus
3. Test all functionality using only keyboard (Enter, Space, Arrow keys)
4. Ensure logical tab order (left-to-right, top-to-bottom)
5. Test skip links functionality
6. Verify focus indicators are visible and clear

### Expected Results:
- All interactive elements are keyboard accessible
- Tab order is logical and predictable
- All functionality works with keyboard only
- Focus indicators are clear and visible
- Skip links work properly

## Screen Reader Testing

### Test Steps:
1. Use NVDA (Windows) or VoiceOver (Mac) to navigate the page
2. Listen to how content is announced
3. Test navigation with screen reader shortcuts
4. Verify alternative text for images
5. Check that ARIA labels and roles are properly announced
6. Test form labels and error messages

### Expected Results:
- Content is announced in logical order
- Alternative text is meaningful
- ARIA attributes are properly utilized
- Form elements have proper labels
- Error messages are clearly announced

## Color and Contrast Testing

### Test Steps:
1. Use tools like WAVE or axe to check contrast ratios
2. Verify no information is conveyed by color alone
3. Test with color blindness simulators
4. Check that focus indicators have sufficient contrast

### Expected Results:
- All text has 4.5:1 contrast ratio (3:1 for large text)
- No information relies solely on color
- Focus indicators are visible against all backgrounds

## Zoom and Magnification Testing

### Test Steps:
1. Zoom to 200% and verify content remains usable
2. Test horizontal scrolling requirements
3. Ensure no horizontal scrolling at 320px width
4. Verify text remains readable and controls usable

### Expected Results:
- Content remains functional at 200% zoom
- No horizontal scrolling needed for 320px width
- Text remains readable and controls accessible
```

## Performance Testing

### Load Testing

```typescript
// tests/performance/load.test.ts
import axios from 'axios';

describe('Performance Tests', () => {
  const BASE_URL = 'http://localhost:3000';
  const CONCURRENT_USERS = 10;
  const TEST_DURATION = 30000; // 30 seconds

  it('handles concurrent users gracefully', async () => {
    const startTime = Date.now();
    const requests = [];

    // Simulate concurrent users accessing different pages
    for (let i = 0; i < CONCURRENT_USERS; i++) {
      const pagePaths = [
        '/',
        '/docs/week-01-02-introduction/foundations-of-physical-ai',
        '/docs/week-03-05-ros2/ros2-architecture-concepts',
        '/docs/week-06-07-simulation/gazebo-simulation-setup'
      ];

      const randomPath = pagePaths[Math.floor(Math.random() * pagePaths.length)];
      requests.push(axios.get(`${BASE_URL}${randomPath}`));
    }

    const responses = await Promise.all(requests);
    const endTime = Date.now();

    // Check that all requests succeeded
    responses.forEach((response, index) => {
      expect(response.status).toBe(200);
    });

    // Check response time
    const totalTime = endTime - startTime;
    expect(totalTime).toBeLessThan(TEST_DURATION);

    console.log(`Concurrent user test completed in ${totalTime}ms`);
  });

  it('measures page load times', async () => {
    const pagesToTest = [
      { path: '/', name: 'Home Page' },
      { path: '/docs/week-01-02-introduction/foundations-of-physical-ai', name: 'Week 1 Lesson' },
      { path: '/docs/week-03-05-ros2/ros2-architecture-concepts', name: 'Week 3 Lesson' },
      { path: '/docs/projects/ros2-package-dev', name: 'Project Page' }
    ];

    for (const page of pagesToTest) {
      const startTime = Date.now();
      const response = await axios.get(`${BASE_URL}${page.path}`, {
        timeout: 10000 // 10 second timeout
      });
      const endTime = Date.now();

      const loadTime = endTime - startTime;

      // Assert that page loads within 3 seconds
      expect(loadTime).toBeLessThan(3000);

      console.log(`${page.name} loaded in ${loadTime}ms with status ${response.status}`);
    }
  });
});
```

## Quality Assurance Processes

### Code Review Checklist

Create a comprehensive code review checklist:

```markdown
# Code Review Checklist

## Content Quality
- [ ] Technical accuracy verified
- [ ] Clear and concise explanations
- [ ] Appropriate difficulty level
- [ ] Consistent terminology
- [ ] Proper learning objectives
- [ ] Adequate examples and exercises

## Accessibility
- [ ] Proper heading hierarchy
- [ ] Alternative text for images
- [ ] Sufficient color contrast
- [ ] Keyboard navigation support
- [ ] ARIA attributes where needed
- [ ] Screen reader compatibility

## Code Quality
- [ ] Clean, readable code
- [ ] Proper error handling
- [ ] Security considerations
- [ ] Performance optimizations
- [ ] Follows style guide
- [ ] Proper documentation

## Testing
- [ ] Unit tests cover main functionality
- [ ] Edge cases handled
- [ ] Integration tests for components
- [ ] Accessibility tests included
- [ ] Performance tests added
- [ ] End-to-end tests cover critical paths

## Documentation
- [ ] Code comments where needed
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Configuration documented
- [ ] Deployment procedures documented
- [ ] Troubleshooting guide updated
```

### Automated Quality Checks

Set up automated quality checks in CI/CD:

```yaml
# .github/workflows/quality-check.yml
name: Quality Assurance

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Run ESLint
        run: |
          cd docs
          npx eslint src/ --ext .ts,.tsx,.js,.jsx --max-warnings 0

      - name: Run Prettier
        run: |
          cd docs
          npx prettier --check "**/*.{js,jsx,ts,tsx,md,mdx,json,yml,yaml}"

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Run TypeScript compiler
        run: |
          cd docs
          npx tsc --noEmit

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Run unit tests
        run: |
          cd docs
          npm test -- --coverage --passWithNoTests

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./docs/coverage/lcov.info
          flags: unittests
          name: codecov-umbrella

  accessibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: |
          cd docs
          npm ci

      - name: Run accessibility tests
        run: |
          cd docs
          npm run test:accessibility

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run security audit
        run: |
          cd docs
          npm audit --audit-level high
```

## Monitoring and Analytics

### Performance Monitoring

Set up performance monitoring:

```typescript
// src/utils/performanceMonitoring.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export interface PerformanceMetrics {
  cls: number; // Cumulative Layout Shift
  fid: number; // First Input Delay
  fcp: number; // First Contentful Paint
  lcp: number; // Largest Contentful Paint
  ttfb: number; // Time to First Byte
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    cls: 0,
    fid: 0,
    fcp: 0,
    lcp: 0,
    ttfb: 0
  };

  private callbacks: Array<(metrics: PerformanceMetrics) => void> = [];

  constructor() {
    this.initializeMonitoring();
  }

  private initializeMonitoring() {
    getCLS(this.updateMetric.bind(this, 'cls'));
    getFID(this.updateMetric.bind(this, 'fid'));
    getFCP(this.updateMetric.bind(this, 'fcp'));
    getLCP(this.updateMetric.bind(this, 'lcp'));
    getTTFB(this.updateMetric.bind(this, 'ttfb'));
  }

  private updateMetric(metricName: keyof PerformanceMetrics, metric: any) {
    this.metrics[metricName] = metric.value;

    // Notify all callbacks
    this.callbacks.forEach(callback => callback(this.metrics));
  }

  public subscribe(callback: (metrics: PerformanceMetrics) => void) {
    this.callbacks.push(callback);

    // Immediately provide current metrics
    callback(this.metrics);
  }

  public unsubscribe(callback: (metrics: PerformanceMetrics) => void) {
    const index = this.callbacks.indexOf(callback);
    if (index > -1) {
      this.callbacks.splice(index, 1);
    }
  }

  public getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  public getMetricsReport(): string {
    const { cls, fid, fcp, lcp, ttfb } = this.metrics;

    return `
Performance Report:
- CLS (Layout Shift): ${cls.toFixed(3)} ${cls > 0.1 ? '⚠️ HIGH' : '✅ OK'}
- FID (Input Delay): ${Math.round(fid)}ms ${fid > 100 ? '⚠️ HIGH' : '✅ OK'}
- FCP (Content Paint): ${Math.round(fcp)}ms ${fcp > 1800 ? '⚠️ SLOW' : '✅ OK'}
- LCP (Largest Paint): ${Math.round(lcp)}ms ${lcp > 2500 ? '⚠️ SLOW' : '✅ OK'}
- TTFB (First Byte): ${Math.round(ttfb)}ms ${ttfb > 600 ? '⚠️ SLOW' : '✅ OK'}
    `.trim();
  }
}

// Initialize performance monitoring
export const performanceMonitor = new PerformanceMonitor();
```

## Test Coverage and Reporting

### Coverage Configuration

Configure test coverage reporting:

```json
// docs/package.json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:accessibility": "jest --testPathPattern=accessibility",
    "test:e2e": "cypress run",
    "test:performance": "jest --testPathPattern=performance"
  },
  "jest": {
    "collectCoverageFrom": [
      "src/**/*.{js,jsx,ts,tsx}",
      "!src/**/*.d.ts",
      "!src/**/node_modules/**"
    ],
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
      }
    },
    "moduleNameMapper": {
      "\\.(css|less|scss|sass)$": "identity-obj-proxy"
    }
  }
}
```

## Summary

A comprehensive testing and quality assurance strategy for the Physical AI & Humanoid Robotics textbook includes:

1. **Unit Testing**: Test individual components and functions
2. **Integration Testing**: Test how components work together
3. **End-to-End Testing**: Test complete user workflows
4. **Accessibility Testing**: Ensure content is accessible to all users
5. **Performance Testing**: Verify the platform performs well
6. **Automated Quality Checks**: Implement CI/CD pipelines
7. **Manual Testing Procedures**: Document manual testing processes
8. **Monitoring and Analytics**: Track performance and usage
9. **Code Review Processes**: Establish quality gates
10. **Documentation**: Maintain testing documentation

By implementing these testing strategies, we ensure that the Physical AI & Humanoid Robotics textbook maintains high quality, reliability, and usability for all learners.