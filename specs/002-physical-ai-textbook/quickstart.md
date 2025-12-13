# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Overview
This guide will help you set up and run the Physical AI & Humanoid Robotics textbook locally, and understand the project structure for contributing content.

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git for version control

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Navigate to the Docusaurus Project
```bash
cd docs
```

### 3. Install Dependencies
```bash
npm install
# or
yarn install
```

### 4. Start Development Server
```bash
npm run start
# or
yarn start
```

This will start a local development server at `http://localhost:3000` with live reloading.

## Project Structure

### Content Organization
```
docs/
├── docs/                 # Course content organized by weeks
│   ├── week-01-02-introduction/    # Weeks 1-2 content
│   ├── week-03-05-ros2/            # Weeks 3-5 content
│   ├── week-06-07-simulation/      # Weeks 6-7 content
│   ├── week-08-10-isaac/           # Weeks 8-10 content
│   ├── week-11-12-humanoid/        # Weeks 11-12 content
│   └── week-13-conversational/     # Week 13 content
├── projects/             # Assessment projects and capstone
├── src/                  # Custom components and pages
├── static/               # Static assets (images, downloads)
├── docusaurus.config.js  # Main configuration
├── sidebars.js           # Navigation structure
└── package.json          # Dependencies and scripts
```

### Week Directory Structure
Each week directory follows this pattern:
```
week-01-02-introduction/
├── foundations-of-physical-ai.md
├── from-digital-ai-to-physical-systems.md
├── overview-humanoid-robotics.md
└── _category_.json       # Category configuration
```

## Adding New Content

### 1. Create a New Week Section
Create a new directory under `docs/` with a descriptive name:
```bash
mkdir docs/week-XX-YY-topic
```

### 2. Add Content Files
Create markdown files for each topic within the week directory:
```bash
touch docs/week-XX-YY-topic/topic-name.md
```

### 3. Configure the Category
Create a `_category_.json` file in the week directory:
```json
{
  "label": "Week XX-YY: Topic Title",
  "position": 1,
  "link": {
    "type": "generated-index",
    "description": "Brief description of what's covered in these weeks."
  }
}
```

### 4. Update Navigation
Add the new week to `sidebars.js` to make it appear in the navigation:
```javascript
module.exports = {
  textbook: [
    // ... existing content
    {
      type: 'category',
      label: 'Week XX-YY: Topic Title',
      items: ['week-XX-YY-topic/topic-name'],
    },
  ],
};
```

## Content Formatting Guidelines

### Markdown Structure
Each content file should follow this structure:
```markdown
---
title: Title of the Content
description: Brief description
tags: [tag1, tag2]
---

# Title of the Content

## Learning Objectives
- Objective 1
- Objective 2

## Introduction
Content introduction...

## Main Content
Detailed content...

## Summary
Key takeaways...

## Next Steps
What to learn next...
```

### Code Blocks
Use appropriate language identifiers:
```python
# Python code for ROS 2 examples
import rclpy
from rclpy.node import Node
```

```bash
# Terminal commands
ros2 run package_name executable_name
```

### Adding Interactive Elements
To add custom components:
```markdown
import ComponentName from '@site/src/components/ComponentName';

<ComponentName prop="value" />
```

## Assessment Projects

### Project Structure
Each project should be placed in the `projects/` directory:
```
projects/
└── project-name/
    ├── README.md
    ├── instructions.md
    ├── requirements.md
    ├── deliverables.md
    └── resources/
        ├── template.py
        └── sample_data.json
```

### Project README Template
```markdown
# Project Name

## Overview
Brief description of the project.

## Learning Objectives
- Objective 1
- Objective 2

## Prerequisites
- Prerequisite 1
- Prerequisite 2

## Requirements
Detailed requirements...

## Deliverables
What students need to submit...

## Evaluation Criteria
How the project will be graded...
```

## Building for Production

### 1. Build the Static Site
```bash
npm run build
# or
yarn build
```

### 2. Serve the Build Locally (for testing)
```bash
npm run serve
# or
yarn serve
```

### 3. Deploy to GitHub Pages
The site is configured to deploy automatically via GitHub Actions when changes are pushed to the main branch.

## Custom Components

### Interactive Demos
Use the InteractiveDemo component for interactive content:
```markdown
import InteractiveDemo from '@site/src/components/InteractiveDemo';

<InteractiveDemo
  title="Demo Title"
  description="Brief description"
  code={`# Python code here`}
/>
```

### Code Runners
Use the CodeRunner component for executable code examples:
```markdown
import CodeRunner from '@site/src/components/CodeRunner';

<CodeRunner
  language="python"
  code={`print("Hello, ROS 2!")`}
/>
```

### Assessment Components
Use the Assessment component for quizzes and exercises:
```markdown
import Assessment from '@site/src/components/Assessment';

<Assessment
  question="What is the primary purpose of ROS 2?"
  options={["Robot Operating System", "Robot Operating System 2", "Really Old System", "Robot Optimization Suite"]}
  answer={1}
/>
```

## Testing

### Run Unit Tests
```bash
npm run test
# or
yarn test
```

### Run E2E Tests
```bash
npm run test:e2e
# or
yarn test:e2e
```

### Lint Content
```bash
npm run lint:content
# or
yarn lint:content
```

## Internationalization

### Adding Translations
1. Create a new directory under `i18n/{locale}/docusaurus-plugin-content-docs/current/`
2. Copy the content structure from the default locale
3. Translate the content

### Language Configuration
Languages are configured in `docusaurus.config.js`:
```javascript
module.exports = {
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'], // English and Urdu
  },
};
```

## Troubleshooting

### Common Issues

#### Port Already in Use
If you get a port in use error:
```bash
# Kill processes on port 3000
lsof -ti:3000 | xargs kill -9
# Then restart
npm run start
```

#### Dependency Issues
If you encounter dependency issues:
```bash
rm -rf node_modules package-lock.json
npm install
```

#### Build Errors
If the build fails, check for:
- Invalid markdown syntax
- Broken links
- Missing image files

## Next Steps

1. Review the [Data Model](./data-model.md) to understand the content structure
2. Check the [Implementation Plan](./plan.md) for the overall architecture
3. Look at existing content examples to understand the format
4. Create your first content page following the guidelines above