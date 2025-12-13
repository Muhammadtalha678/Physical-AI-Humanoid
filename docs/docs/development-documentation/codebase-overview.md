---
title: Codebase Overview and Developer Guide
sidebar_position: 1
description: Comprehensive documentation for the Physical AI & Humanoid Robotics textbook codebase
duration: 120
difficulty: intermediate
learning_objectives:
  - Understand the overall structure of the textbook codebase
  - Learn how to contribute to and extend the textbook content
  - Master the development workflow for adding new content
  - Understand the technical architecture and dependencies
---

# Codebase Overview and Developer Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Navigate and understand the structure of the Physical AI & Humanoid Robotics textbook codebase
- Contribute new content following established patterns and standards
- Set up the development environment for local textbook development
- Understand the technical architecture and dependencies
- Follow the established development workflow

## Project Structure

The Physical AI & Humanoid Robotics textbook project follows a well-organized structure:

```
physical-ai-textbook/
├── docs/                           # Docusaurus documentation site
│   ├── blog/                       # Blog posts (if any)
│   ├── docs/                       # Main textbook content
│   │   ├── week-01-02-introduction/    # Weeks 1-2 content
│   │   ├── week-03-05-ros2/          # Weeks 3-5 content
│   │   ├── week-06-07-simulation/    # Weeks 6-7 content
│   │   ├── week-08-10-isaac/         # Weeks 8-10 content
│   │   ├── week-11-12-humanoid/      # Weeks 11-12 content
│   │   ├── week-13-conversational/   # Week 13 content
│   │   ├── projects/                 # Project-based learning content
│   │   │   ├── ros2-package-dev/     # ROS 2 project
│   │   │   ├── gazebo-simulation-impl/ # Gazebo project
│   │   │   ├── isaac-perception-pipeline/ # Isaac project
│   │   │   └── capstone-autonomous-humanoid/ # Capstone project
│   │   └── development-documentation/ # Development guides
│   ├── src/                        # Custom React components
│   │   ├── components/             # Reusable UI components
│   │   │   ├── InteractiveDemo/    # Interactive code demo component
│   │   │   ├── CodeRunner/         # Code execution component
│   │   │   ├── Assessment/         # Assessment component
│   │   │   └── AssessmentTracker/  # Progress tracking component
│   │   └── css/                    # Custom styles
│   ├── static/                     # Static assets
│   │   └── resources/              # Project resources and templates
│   ├── docusaurus.config.ts        # Main Docusaurus configuration
│   ├── sidebars.ts                 # Navigation sidebar configuration
│   └── package.json                # Node.js dependencies
├── specs/                          # Feature specifications
│   └── 002-physical-ai-textbook/   # Main textbook spec
├── .github/                        # GitHub configuration
├── .claude/                        # Claude Code configuration
└── README.md                       # Project overview
```

## Core Architecture

### Frontend Framework: Docusaurus 3.x

The textbook is built using Docusaurus 3.x, a modern static site generator optimized for documentation:

- **Markdown-based content**: All textbook content is written in Markdown with frontmatter metadata
- **React components**: Custom interactive components are built with React
- **TypeScript**: Strong typing for component development
- **MDX support**: Ability to embed React components directly in Markdown

### Content Organization

The content is organized by the 13-week curriculum structure:

1. **Week-based directories**: Each week has its own directory with related content
2. **Project-focused sections**: Separate project content integrated with theoretical content
3. **Progressive complexity**: Content builds in complexity from week to week
4. **Cross-references**: Links and references between related topics

### Custom Components

The textbook includes several custom React components for enhanced interactivity:

#### InteractiveDemo Component

```tsx
// src/components/InteractiveDemo/InteractiveDemo.tsx
import React, { useState } from 'react';
import styles from './InteractiveDemo.module.css';

interface InteractiveDemoProps {
  defaultCode?: string;
  language?: string;
  title?: string;
}

const InteractiveDemo: React.FC<InteractiveDemoProps> = ({
  defaultCode = '// Write your code here',
  language = 'python',
  title = 'Interactive Demo'
}) => {
  const [code, setCode] = useState(defaultCode);
  const [output, setOutput] = useState('');

  const executeCode = () => {
    // Code execution logic (simulated)
    setOutput('Code executed successfully!');
  };

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>{title}</h3>
      <div className={styles.editor}>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className={styles.codeInput}
          rows={10}
        />
      </div>
      <div className={styles.controls}>
        <button onClick={executeCode} className={styles.executeBtn}>
          Run Code
        </button>
        <button onClick={() => setCode(defaultCode)} className={styles.resetBtn}>
          Reset
        </button>
      </div>
      <div className={styles.output}>
        <h4>Output:</h4>
        <pre>{output}</pre>
      </div>
    </div>
  );
};

export default InteractiveDemo;
```

#### Assessment Component

```tsx
// src/components/Assessment/Assessment.tsx
import React, { useState } from 'react';
import styles from './Assessment.module.css';

interface Option {
  text: string;
  explanation?: string;
}

interface AssessmentProps {
  question: string;
  type: 'multiple-choice' | 'single-choice';
  options: string[];
  correctIndex: number;
  explanation: string;
}

const Assessment: React.FC<AssessmentProps> = ({
  question,
  type,
  options,
  correctIndex,
  explanation
}) => {
  const [selected, setSelected] = useState<number | number[]>(type === 'single-choice' ? -1 : []);
  const [submitted, setSubmitted] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleOptionSelect = (index: number) => {
    if (type === 'single-choice') {
      setSelected(index);
    } else {
      setSelected(prev =>
        Array.isArray(prev)
          ? prev.includes(index)
            ? prev.filter(i => i !== index)
            : [...prev, index]
          : [index]
      );
    }
  };

  const handleSubmit = () => {
    setSubmitted(true);
    setShowExplanation(true);
  };

  const isCorrect = type === 'single-choice'
    ? selected === correctIndex
    : Array.isArray(selected) && selected.length === 1 && selected[0] === correctIndex;

  return (
    <div className={styles.assessmentContainer}>
      <div className={styles.question}>{question}</div>

      <div className={styles.options}>
        {options.map((option, index) => (
          <div
            key={index}
            className={`${styles.option} ${
              submitted
                ? index === correctIndex
                  ? styles.correctOption
                  : Array.isArray(selected)
                    ? selected.includes(index) && !isCorrect
                      ? styles.incorrectOption
                      : ''
                    : selected === index && !isCorrect
                      ? styles.incorrectOption
                      : ''
                : (type === 'single-choice' ? selected === index : Array.isArray(selected) && selected.includes(index))
                  ? styles.selectedOption
                  : ''
            }`}
            onClick={() => !submitted && handleOptionSelect(index)}
          >
            <input
              type={type === 'single-choice' ? 'radio' : 'checkbox'}
              checked={type === 'single-choice' ? selected === index : Array.isArray(selected) && selected.includes(index)}
              onChange={() => {}}
              disabled={submitted}
              className={styles.optionInput}
            />
            <span className={styles.optionText}>{option}</span>
          </div>
        ))}
      </div>

      {!submitted ? (
        <button
          className={styles.submitBtn}
          onClick={handleSubmit}
          disabled={type === 'single-choice' ? selected === -1 : (Array.isArray(selected) && selected.length === 0)}
        >
          Submit Answer
        </button>
      ) : (
        <div className={`${styles.result} ${isCorrect ? styles.correct : styles.incorrect}`}>
          {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
        </div>
      )}

      {showExplanation && (
        <div className={styles.explanation}>
          <strong>Explanation:</strong> {explanation}
        </div>
      )}
    </div>
  );
};

export default Assessment;
```

## Development Environment Setup

### Prerequisites

Before contributing to the textbook, ensure you have the following installed:

- **Node.js**: Version 18.0 or higher
- **npm or yarn**: Package manager (npm comes with Node.js)
- **Git**: Version control system
- **A code editor**: VS Code recommended with TypeScript extensions

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd physical-ai-textbook
   ```

2. **Install dependencies**:
   ```bash
   cd docs
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```
   This will start a local server at http://localhost:3000

### Development Workflow

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** to the appropriate files

3. **Test your changes** locally:
   ```bash
   npm start
   ```

4. **Build the site** to ensure it works correctly:
   ```bash
   npm run build
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

6. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Content Creation Guidelines

### Adding New Content

When adding new content to the textbook, follow these guidelines:

#### File Structure

Create content files in the appropriate week directory:

```
docs/docs/week-XX-YY-topic/
├── topic-overview.md
├── subtopic-1.md
└── _category_.json
```

#### Markdown Frontmatter

Each content file should include proper frontmatter:

```markdown
---
title: Your Content Title
sidebar_position: 1
description: Brief description of the content
duration: 60  # Estimated time to complete in minutes
difficulty: intermediate  # beginner | intermediate | advanced
learning_objectives:
  - Objective 1
  - Objective 2
  - Objective 3
---

# Your Content Title

## Learning Objectives

By the end of this section, you will be able to:
- Objective 1
- Objective 2
- Objective 3

## Main Content

Your content goes here...
```

#### Using Components

You can use custom components in your Markdown files:

```markdown
<Assessment
  question="What is the primary function of a ROS node?"
  type="multiple-choice"
  options={[
    "To store robot configuration",
    "To perform computation and communicate with other nodes",
    "To manage hardware components only",
    "To provide user interfaces"
  ]}
  correctIndex={1}
  explanation="A ROS node is a process that performs computation and communicates with other nodes through topics, services, and actions."
/>
```

### Code Examples

When including code examples, use proper syntax highlighting:

```python
# Python code example
def example_function():
    """Example function with docstring."""
    return "Hello, Physical AI!"
```

### Images and Assets

Place images in the `static/` directory and reference them using absolute paths:

```markdown
![Description of image](/img/image-name.png)
```

## Technical Architecture

### Docusaurus Configuration

The main configuration file `docusaurus.config.ts` defines:

- Site metadata (title, description, URL)
- Theme configuration
- Plugin configuration
- Deployment settings
- Internationalization settings

### Navigation Structure

The `sidebars.ts` file defines the navigation structure:

```typescript
// sidebars.ts
const sidebars = {
  textbook: [
    {
      type: 'category',
      label: 'Week 1-2: Introduction to Physical AI',
      items: [
        'week-01-02-introduction/foundations-of-physical-ai',
        'week-01-02-introduction/from-digital-ai-to-physical-systems',
        'week-01-02-introduction/overview-humanoid-robotics',
      ],
    },
    // ... more categories
  ],
};

export default sidebars;
```

### Custom Styling

Custom styles are defined in `src/css/custom.css` and component-specific styles use CSS modules.

## Testing and Quality Assurance

### Local Testing

Before submitting changes:

1. Test the development server: `npm start`
2. Verify all links work correctly
3. Check that custom components render properly
4. Ensure code examples are correct
5. Validate that the build process completes: `npm run build`

### Content Quality Standards

- **Accuracy**: All technical information must be correct
- **Clarity**: Use clear, concise language appropriate for the target audience
- **Consistency**: Follow established formatting and terminology
- **Completeness**: Include all necessary information for understanding
- **Engagement**: Include interactive elements and practical examples

## Deployment Configuration

### GitHub Pages Deployment

The textbook is deployed to GitHub Pages. The deployment configuration includes:

- **Build command**: `npm run build`
- **Publish directory**: `build/`
- **Custom domain**: If applicable

### Production Build

To create a production build:

```bash
npm run build
```

This creates an optimized static site in the `build/` directory.

## Performance Optimization

### Image Optimization

- Use appropriate formats (WebP for complex images, SVG for simple graphics)
- Compress images before adding to the repository
- Use lazy loading for images outside the initial viewport

### Code Splitting

Docusaurus automatically handles code splitting for:
- Individual pages
- Documentation sections
- Heavy components

### Caching Strategy

- Leverage browser caching for static assets
- Use appropriate cache headers
- Implement service workers for offline access

## Accessibility Features

The textbook includes accessibility features:

- Semantic HTML structure
- Proper heading hierarchy
- Alt text for images
- Keyboard navigation support
- Screen reader compatibility
- Sufficient color contrast

## Analytics and Monitoring

### User Progress Tracking

The system tracks:
- Content completion rates
- Time spent on pages
- Assessment performance
- Navigation patterns

### Performance Monitoring

Monitor:
- Page load times
- Build times
- Error rates
- User engagement metrics

## Troubleshooting Common Issues

### Build Issues

**Problem**: `npm run build` fails with module errors
**Solution**: Clear cache and reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Component Issues

**Problem**: Custom components don't render
**Solution**: Check import paths and ensure component files exist

### Navigation Issues

**Problem**: Pages don't appear in sidebar
**Solution**: Verify the page is referenced in `sidebars.ts` and has correct frontmatter

## Contribution Best Practices

1. **Follow the style guide**: Maintain consistency with existing content
2. **Test thoroughly**: Verify all functionality before submitting
3. **Write clear documentation**: Include explanations for complex concepts
4. **Use meaningful commit messages**: Follow conventional commit format
5. **Keep pull requests focused**: Address one topic per PR when possible

## Getting Help

- **Documentation**: Refer to the official Docusaurus documentation
- **Code examples**: Look at existing content for patterns
- **Community**: Reach out to the development team for complex issues

This guide provides the foundation for contributing to and maintaining the Physical AI & Humanoid Robotics textbook. Following these guidelines ensures consistency, quality, and maintainability of the educational content.