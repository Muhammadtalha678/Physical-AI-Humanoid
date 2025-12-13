---
title: Accessibility Guide
sidebar_position: 3
description: Ensuring the textbook is accessible to all learners including those with disabilities
duration: 120
difficulty: intermediate
learning_objectives:
  - Implement accessibility features for inclusive learning
  - Follow WCAG guidelines for educational content
  - Use semantic HTML and ARIA attributes appropriately
  - Test and validate accessibility compliance
---

# Accessibility Guide

## Learning Objectives

By the end of this guide, you will be able to:
- Implement accessibility features that make the textbook usable for all learners
- Follow WCAG 2.1 AA guidelines for educational content
- Use semantic HTML and ARIA attributes to enhance accessibility
- Test and validate accessibility compliance using automated and manual methods
- Create inclusive content that accommodates various disabilities

## Introduction to Web Accessibility

### What is Web Accessibility?

Web accessibility means that websites, tools, and technologies are designed and developed so that people with disabilities can use them. For the Physical AI & Humanoid Robotics textbook, this means ensuring that:

- People with visual impairments can access content using screen readers
- People with hearing impairments can understand audio content through transcripts
- People with motor impairments can navigate using keyboard-only interactions
- People with cognitive disabilities can understand and navigate content easily

### Legal and Ethical Considerations

Web accessibility is not just good practice—it's often required by law:

- **Section 508**: U.S. federal agencies must make electronic and information technology accessible
- **ADA**: The Americans with Disabilities Act applies to websites
- **EU Web Accessibility Directive**: EU countries must make public sector websites accessible
- **Educational requirements**: Many educational institutions have accessibility policies

## WCAG Guidelines

### Understanding WCAG 2.1

The Web Content Accessibility Guidelines (WCAG) 2.1 provide a framework for accessible web content with four principles:

#### 1. Perceivable
Information and user interface components must be presentable to users in ways they can perceive.

**Examples for our textbook:**
- Providing text alternatives for images
- Creating captions for videos
- Using sufficient color contrast
- Making content adaptable to different formats

#### 2. Operable
User interface components and navigation must be operable.

**Examples for our textbook:**
- All functionality available from a keyboard
- Enough time for users to read and use content
- Content that doesn't cause seizures
- Easy navigation and identification of content

#### 3. Understandable
Information and the operation of user interface must be understandable.

**Examples for our textbook:**
- Text content readable and comprehensible
- Interface components predictable in operation
- Input assistance provided when users make mistakes

#### 4. Robust
Content must be robust enough that it can be interpreted reliably by a wide variety of user agents, including assistive technologies.

**Examples for our textbook:**
- Compatible with current and future assistive technologies
- Valid HTML that can be parsed by different browsers

## Semantic HTML and Structure

### Proper Heading Hierarchy

Use proper heading levels to create a logical content structure:

```markdown
<!-- DO: Use proper heading hierarchy -->
# Week 1: Introduction to Physical AI
## Learning Objectives
### Understanding Physical AI Fundamentals
#### Key Principles
## Course Overview
### What You'll Learn

<!-- DON'T: Skip heading levels -->
# Week 1: Introduction to Physical AI
#### This skips heading level 2 and 3
```

### Semantic HTML Elements

Use semantic HTML elements to provide meaning to content:

```html
<!-- Use semantic elements appropriately -->
<header>
  <nav aria-label="Main navigation">
    <!-- Navigation links -->
  </nav>
</header>

<main>
  <article>
    <section>
      <h2>Section Title</h2>
      <p>Content paragraph</p>
    </section>
  </article>
</main>

<aside aria-label="Supplementary information">
  <!-- Side notes, additional information -->
</aside>

<footer>
  <!-- Footer content -->
</footer>
```

### Landmark Roles

Use landmark roles to help screen reader users navigate:

```html
<!-- Docusaurus automatically provides many landmarks, but ensure custom components use them -->
<div role="banner">Header content</div>
<div role="main" id="main">Main content</div>
<div role="complementary">Sidebar content</div>
<div role="contentinfo">Footer content</div>
```

## Text Alternatives and Media

### Image Accessibility

Provide meaningful alternative text for images:

```markdown
<!-- Good: Descriptive alt text -->
![A robot arm picking up a colored cube](/img/robot-arm-grasping.jpg)

<!-- Good: Decorative images (empty alt) -->
![Purely decorative element](/img/decoration.jpg)

<!-- Good: Functional images -->
![Click to expand section](/img/expand-icon.jpg)

<!-- Bad: Generic alt text -->
![Picture](/img/robot-arm.jpg)
```

### Video and Audio Content

Provide alternatives for time-based media:

```html
<!-- Video with captions and transcript -->
<video controls>
  <source src="/videos/lecture.mp4" type="video/mp4">
  <track kind="captions" src="/videos/lecture-en.vtt" srclang="en" label="English">
  <track kind="descriptions" src="/videos/lecture-desc.vtt" srclang="en" label="Audio descriptions">
  <p>Video player not supported. <a href="/transcripts/lecture.txt">Read transcript</a>.</p>
</video>

<!-- Audio with transcript -->
<audio controls>
  <source src="/audio/lecture.mp3" type="audio/mpeg">
  <p>Audio player not supported. <a href="/transcripts/audio-lecture.txt">Read transcript</a>.</p>
</audio>
```

### Complex Images

Provide detailed descriptions for complex images like diagrams:

```markdown
<details>
  <summary>Image Description: ROS 2 Architecture Diagram</summary>
  <p>The diagram shows the ROS 2 architecture with the following components:</p>
  <ul>
    <li>At the bottom, the Operating System layer (Ubuntu/Linux)</li>
    <li>In the middle, the ROS Client Libraries (rclcpp, rclpy)</li>
    <li>At the top, various nodes connected by topics</li>
    <li>Nodes include: perception_node, navigation_node, control_node</li>
  </ul>
</details>

![ROS 2 Architecture](/img/ros2-architecture.svg)
```

## Keyboard Navigation

### Focus Management

Ensure all interactive elements are keyboard accessible:

```tsx
// Example of proper focus management in a custom component
import React, { useRef, useEffect } from 'react';

const InteractiveDemo: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Manage focus programmatically when needed
  useEffect(() => {
    if (containerRef.current) {
      // Ensure the container can receive focus
      containerRef.current.tabIndex = 0;
    }
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Handle keyboard shortcuts appropriately
    if (e.key === 'Enter' || e.key === ' ') {
      // Trigger the same action as a click
      buttonRef.current?.click();
    }
  };

  return (
    <div
      ref={containerRef}
      onKeyDown={handleKeyDown}
      className="interactive-demo"
      tabIndex={-1} // Allow programmatic focus
    >
      <button ref={buttonRef} className="action-button">
        Run Simulation
      </button>
    </div>
  );
};
```

### Skip Links

Provide skip links to help users bypass repetitive navigation:

```html
<!-- This is handled automatically by Docusaurus, but can be customized -->
<a href="#main-content" class="skip-link">Skip to main content</a>
<a href="#toc" class="skip-link">Skip to table of contents</a>
```

## Color and Visual Design

### Color Contrast

Maintain sufficient color contrast ratios:

```css
/* WCAG AA requires 4.5:1 for normal text, 3:1 for large text */
.accessible-text {
  color: #000000; /* Black */
  background-color: #ffffff; /* White */
  /* Contrast ratio: 21:1 (Excellent) */
}

.good-contrast {
  color: #212121; /* Dark gray */
  background-color: #f5f5f5; /* Light gray */
  /* Contrast ratio: 15.1:1 (Excellent) */
}

/* Avoid low contrast combinations */
.poor-contrast {
  color: #757575; /* Medium gray */
  background-color: #bdbdbd; /* Light medium gray */
  /* Contrast ratio: 1.7:1 (Poor) */
}
```

### Color Independence

Don't rely solely on color to convey information:

```html
<!-- Good: Color + text/pattern -->
<div class="status-indicator success">
  ✓ Success
</div>
<div class="status-indicator error">
  ✗ Error
</div>

<!-- Bad: Color alone -->
<div class="status-indicator" style="background-color: green;">
  Success
</div>
<div class="status-indicator" style="background-color: red;">
  Error
</div>
```

### Focus Indicators

Provide clear focus indicators:

```css
/* Ensure focus indicators are visible and accessible */
.focusable-element:focus {
  outline: 2px solid #4a90e2;
  outline-offset: 2px;
  /* Never remove outline completely - provide visible alternative */
}

/* For custom focus indicators */
.custom-focus {
  border: 2px solid transparent;
}

.custom-focus:focus {
  border-color: #4a90e2;
  box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.3);
}
```

## Forms and Input Controls

### Labeling

Associate labels with form controls:

```tsx
// Accessible form controls
import React from 'react';

const AccessibleForm: React.FC = () => {
  return (
    <form>
      {/* Good: Explicit label */}
      <label htmlFor="email-input">Email Address</label>
      <input
        type="email"
        id="email-input"
        name="email"
        required
        aria-describedby="email-help"
      />
      <div id="email-help">We'll never share your email.</div>

      {/* Good: Implicit label */}
      <label>
        Preferred Language
        <select name="language" defaultValue="en">
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
        </select>
      </label>

      {/* Good: Fieldset for grouped controls */}
      <fieldset>
        <legend>Programming Experience Level</legend>
        <label>
          <input type="radio" name="experience" value="beginner" /> Beginner
        </label>
        <label>
          <input type="radio" name="experience" value="intermediate" /> Intermediate
        </label>
        <label>
          <input type="radio" name="experience" value="advanced" /> Advanced
        </label>
      </fieldset>
    </form>
  );
};
```

### Error Handling

Provide clear error messages and instructions:

```tsx
// Accessible error handling
import React, { useState } from 'react';

interface FormState {
  email: string;
  experience: string;
  errors: {
    email?: string;
    experience?: string;
  };
}

const AccessibleFormWithErrorHandling: React.FC = () => {
  const [state, setState] = useState<FormState>({
    email: '',
    experience: '',
    errors: {}
  });

  const validateForm = (): boolean => {
    const errors = {};

    if (!state.email) {
      errors['email'] = 'Email address is required';
    } else if (!/\S+@\S+\.\S+/.test(state.email)) {
      errors['email'] = 'Please enter a valid email address';
    }

    if (!state.experience) {
      errors['experience'] = 'Please select your experience level';
    }

    setState(prev => ({ ...prev, errors }));
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      // Submit form
      console.log('Form submitted successfully');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className={`form-field ${state.errors.email ? 'error' : ''}`}>
        <label htmlFor="email">Email Address *</label>
        <input
          type="email"
          id="email"
          value={state.email}
          onChange={(e) => setState(prev => ({ ...prev, email: e.target.value, errors: { ...prev.errors, email: undefined } }))}
          aria-invalid={!!state.errors.email}
          aria-describedby={state.errors.email ? "email-error" : undefined}
        />
        {state.errors.email && (
          <div id="email-error" className="error-message" role="alert">
            {state.errors.email}
          </div>
        )}
      </div>

      <div className={`form-field ${state.errors.experience ? 'error' : ''}`}>
        <fieldset>
          <legend>Experience Level *</legend>
          <div>
            <label>
              <input
                type="radio"
                name="experience"
                value="beginner"
                checked={state.experience === 'beginner'}
                onChange={(e) => setState(prev => ({ ...prev, experience: e.target.value, errors: { ...prev.errors, experience: undefined } }))}
              />
              Beginner
            </label>
          </div>
          <div>
            <label>
              <input
                type="radio"
                name="experience"
                value="intermediate"
                checked={state.experience === 'intermediate'}
                onChange={(e) => setState(prev => ({ ...prev, experience: e.target.value, errors: { ...prev.errors, experience: undefined } }))}
              />
              Intermediate
            </label>
          </div>
          <div>
            <label>
              <input
                type="radio"
                name="experience"
                value="advanced"
                checked={state.experience === 'advanced'}
                onChange={(e) => setState(prev => ({ ...prev, experience: e.target.value, errors: { ...prev.errors, experience: undefined } }))}
              />
              Advanced
            </label>
          </div>
        </fieldset>
        {state.errors.experience && (
          <div className="error-message" role="alert">
            {state.errors.experience}
          </div>
        )}
      </div>

      <button type="submit">Submit</button>
    </form>
  );
};
```

## Custom Components Accessibility

### Interactive Components

Make custom interactive components accessible:

```tsx
// Accessible custom button component
import React from 'react';

interface AccessibleButtonProps {
  children: React.ReactNode;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'tertiary';
  disabled?: boolean;
  ariaLabel?: string;
}

const AccessibleButton: React.FC<AccessibleButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
  ariaLabel
}) => {
  const handleClick = () => {
    if (!disabled) {
      onClick();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <button
      className={`accessible-button button-${variant} ${disabled ? 'disabled' : ''}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      disabled={disabled}
      aria-label={ariaLabel}
      tabIndex={disabled ? -1 : 0}
      type="button"
    >
      {children}
    </button>
  );
};

// Accessible dropdown component
interface AccessibleDropdownProps {
  label: string;
  options: Array<{ value: string; label: string }>;
  onSelect: (value: string) => void;
  defaultValue?: string;
}

const AccessibleDropdown: React.FC<AccessibleDropdownProps> = ({
  label,
  options,
  onSelect,
  defaultValue
}) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const [activeIndex, setActiveIndex] = React.useState(-1);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setIsOpen(!isOpen);
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex(prev => Math.min(prev + 1, options.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex(prev => Math.max(prev - 1, 0));
    }
  };

  const selectOption = (value: string) => {
    onSelect(value);
    setIsOpen(false);
    setActiveIndex(-1);
  };

  return (
    <div className="accessible-dropdown" ref={dropdownRef}>
      <div
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-owns="dropdown-options"
        aria-labelledby="dropdown-label"
        tabIndex={0}
        onKeyDown={handleKeyDown}
        className="dropdown-trigger"
      >
        <span id="dropdown-label">{label}</span>
        <span className="dropdown-arrow">{isOpen ? '▲' : '▼'}</span>
      </div>

      {isOpen && (
        <ul
          id="dropdown-options"
          role="listbox"
          className="dropdown-options"
        >
          {options.map((option, index) => (
            <li
              key={option.value}
              role="option"
              aria-selected={defaultValue === option.value}
              className={`dropdown-option ${index === activeIndex ? 'focused' : ''}`}
              onClick={() => selectOption(option.value)}
              onMouseEnter={() => setActiveIndex(index)}
            >
              {option.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
```

## Assessment Accessibility

### Accessible Assessment Components

Ensure assessment components are fully accessible:

```tsx
// Accessible assessment component
import React, { useState } from 'react';

interface AssessmentOption {
  text: string;
  explanation?: string;
}

interface AccessibleAssessmentProps {
  question: string;
  type: 'single-choice' | 'multiple-choice';
  options: AssessmentOption[];
  correctIndexes: number[]; // For multiple correct answers
  explanation: string;
  onAnswerSubmit?: (isCorrect: boolean, selectedIndexes: number[]) => void;
}

const AccessibleAssessment: React.FC<AccessibleAssessmentProps> = ({
  question,
  type,
  options,
  correctIndexes,
  explanation,
  onAnswerSubmit
}) => {
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const toggleSelection = (index: number) => {
    if (submitted) return;

    if (type === 'single-choice') {
      setSelectedIndexes([index]);
    } else {
      if (selectedIndexes.includes(index)) {
        setSelectedIndexes(selectedIndexes.filter(i => i !== index));
      } else {
        setSelectedIndexes([...selectedIndexes, index]);
      }
    }
  };

  const handleSubmit = () => {
    setSubmitted(true);
    setShowExplanation(true);
    if (onAnswerSubmit) {
      const isCorrect =
        selectedIndexes.length === correctIndexes.length &&
        selectedIndexes.every(index => correctIndexes.includes(index));
      onAnswerSubmit(isCorrect, selectedIndexes);
    }
  };

  const isCorrect =
    selectedIndexes.length === correctIndexes.length &&
    selectedIndexes.every(index => correctIndexes.includes(index));

  return (
    <div className="accessible-assessment" role="region" aria-labelledby="question-text">
      <h3 id="question-text" className="question">{question}</h3>

      <div className="options-container" role="group" aria-label="Answer options">
        {options.map((option, index) => {
          const isSelected = selectedIndexes.includes(index);
          const isCorrectOption = correctIndexes.includes(index);

          return (
            <div
              key={index}
              className={`option ${isSelected ? 'selected' : ''} ${
                submitted
                  ? isCorrectOption ? 'correct' : isSelected ? 'incorrect' : ''
                  : ''
              }`}
              onClick={() => toggleSelection(index)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleSelection(index);
                }
              }}
              role="radio"
              aria-checked={isSelected}
              tabIndex={0}
            >
              <input
                type={type === 'single-choice' ? 'radio' : 'checkbox'}
                checked={isSelected}
                onChange={() => {}}
                readOnly
                className="option-input"
              />
              <span className="option-text">{option.text}</span>
            </div>
          );
        })}
      </div>

      {!submitted ? (
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={selectedIndexes.length === 0}
          aria-disabled={selectedIndexes.length === 0}
        >
          Submit Answer
        </button>
      ) : (
        <div
          className={`result ${isCorrect ? 'correct' : 'incorrect'}`}
          role="status"
          aria-live="polite"
        >
          {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
        </div>
      )}

      {showExplanation && (
        <div className="explanation" role="complementary">
          <h4>Explanation:</h4>
          <p>{explanation}</p>
        </div>
      )}
    </div>
  );
};
```

## Testing and Validation

### Automated Testing Tools

Use automated tools to catch accessibility issues:

```bash
# Install accessibility testing tools
npm install --save-dev axe-core @axe-core/react

# Example of accessibility testing in component
```

```tsx
// Example of integrating accessibility testing
import React from 'react';
import { useEffect } from 'react';
import { configureAxe, checkA11y } from 'jest-axe';

// For development, you can also use react-axe for real-time feedback
if (process.env.NODE_ENV !== 'production') {
  const axe = require('@axe-core/react');
  axe(React, { timeout: 1000, showErrors: true });
}
```

### Manual Testing Checklist

Perform manual testing for comprehensive accessibility:

#### Keyboard Navigation
- [ ] All interactive elements accessible via Tab key
- [ ] Logical tab order (left to right, top to bottom)
- [ ] Visible focus indicators on all interactive elements
- [ ] Skip links work correctly
- [ ] All functionality available via keyboard only

#### Screen Reader Testing
- [ ] Content announced logically by screen readers
- [ ] Alternative text provided for images
- [ ] Form labels properly associated
- [ ] ARIA attributes used appropriately
- [ ] Error messages properly announced

#### Visual Testing
- [ ] Sufficient color contrast (4.5:1 minimum)
- [ ] Information not conveyed by color alone
- [ ] Text remains readable when enlarged to 200%
- [ ] Content remains usable when magnified
- [ ] No content clipped when text is enlarged

### Automated Testing Setup

Set up automated accessibility testing in CI/CD:

```javascript
// jest.config.js
module.exports = {
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.js'],
  testEnvironment: 'jsdom',
  testMatch: ['**/__tests__/**/*.[jt]s?(x)', '**/?(*.)+(spec|test).[jt]s?(x)'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};

// setupTests.js
import '@testing-library/jest-dom';
import { configureAxe } from 'jest-axe';

configureAxe({
  rules: {
    // Disable certain rules that might be overly strict for documentation
    'color-contrast': { enabled: true },
    'heading-order': { enabled: true },
    'duplicate-id': { enabled: true },
  },
});
```

```javascript
// Example accessibility test
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import Assessment from '../components/Assessment/Assessment';

expect.extend(toHaveNoViolations);

describe('Assessment Component', () => {
  it('should have no accessibility violations', async () => {
    const { container } = render(
      <Assessment
        question="What is a ROS node?"
        type="multiple-choice"
        options={['A process', 'A function', 'A variable', 'A class']}
        correctIndex={0}
        explanation="A ROS node is a process that performs computation."
      />
    );

    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
```

## ARIA Attributes

### When and How to Use ARIA

ARIA (Accessible Rich Internet Applications) should be used sparingly and appropriately:

```tsx
// Good use of ARIA
import React from 'react';

const ProgressBar: React.FC<{ value: number; max: number; label: string }> = ({
  value,
  max,
  label
}) => {
  return (
    <div>
      <div id="progress-label">{label}</div>
      <div
        role="progressbar"
        aria-labelledby="progress-label"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        style={{
          width: '100%',
          height: '20px',
          backgroundColor: '#f0f0f0',
          position: 'relative'
        }}
      >
        <div
          style={{
            width: `${(value / max) * 100}%`,
            height: '100%',
            backgroundColor: '#4caf50'
          }}
        />
      </div>
    </div>
  );
};

// Live regions for dynamic content
const Notification: React.FC<{ message: string; isVisible: boolean }> = ({
  message,
  isVisible
}) => {
  return (
    <div>
      <div
        aria-live="polite"
        aria-atomic="true"
        className="notification"
        style={{ display: isVisible ? 'block' : 'none' }}
      >
        {message}
      </div>
    </div>
  );
};
```

### ARIA Best Practices

- Use native HTML elements when possible instead of ARIA
- Don't use ARIA if native semantics already provide the same meaning
- Keep ARIA simple and avoid over-engineering
- Test ARIA implementations with actual assistive technologies

## Inclusive Content Writing

### Plain Language

Write content that is easy to understand:

```markdown
<!-- Good: Plain language -->
The robot uses sensors to understand its environment. Cameras help it see, and LIDAR helps it measure distances.

<!-- Avoid: Jargon without explanation -->
The robotic platform employs multi-modal sensory fusion incorporating computer vision and LiDAR ranging for environmental perception.
```

### Clear Instructions

Provide clear, actionable instructions:

```markdown
<!-- Good: Clear steps -->
1. Open a terminal window
2. Navigate to your ROS workspace: `cd ~/ros_ws`
3. Build your package: `colcon build`
4. Source the setup file: `source install/setup.bash`

<!-- Avoid: Unclear instructions -->
Build your ROS package and source it.
```

## Performance and Accessibility Balance

Optimize for both performance and accessibility:

```tsx
// Balance lazy loading with accessibility
import { useState, useEffect } from 'react';

const AccessibleLazyImage: React.FC<{ src: string; alt: string; caption?: string }> = ({
  src,
  alt,
  caption
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const imgRef = React.useRef<HTMLImageElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1, rootMargin: '50px' }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => {
      if (imgRef.current) {
        observer.unobserve(imgRef.current);
      }
    };
  }, []);

  return (
    <figure>
      <img
        ref={imgRef}
        src={isVisible ? src : 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIi8+'}
        alt={alt}
        onLoad={() => setIsLoaded(true)}
        style={{ opacity: isLoaded || !isVisible ? 1 : 0, transition: 'opacity 0.3s' }}
        loading="lazy"
      />
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
};
```

## Compliance and Standards

### WCAG 2.1 AA Compliance

Ensure compliance with WCAG 2.1 Level AA:

#### Success Criteria Checklist

**1.1 Text Alternatives**
- [ ] All non-text content has appropriate text alternatives
- [ ] Complex images have detailed descriptions
- [ ] CAPTCHAs have alternative forms

**1.2 Time-based Media**
- [ ] Prerecorded audio has transcripts
- [ ] Prerecorded video has captions
- [ ] Live audio has captions
- [ ] Audio description provided for video content

**1.3 Adaptable**
- [ ] Information and relationships are programmatically determinable
- [ ] Content can be presented in different ways
- [ ] Images of text can be visually customized

**1.4 Distinguishable**
- [ ] Color is not the only means of conveying information
- [ ] Sufficient contrast (4.5:1 for normal text, 3:1 for large text)
- [ ] Text can be resized up to 200%
- [ ] Text spacing can be adjusted

**2.1 Keyboard Accessible**
- [ ] All functionality available from keyboard
- [ ] No keyboard traps
- [ ] Sufficient time to complete tasks

**2.2 Enough Time**
- [ ] Users can control time limits
- [ ] Users can control moving content
- [ ] Re-authentication is possible

**2.3 Seizures and Physical Reactions**
- [ ] No content flashes more than 3 times per second

**2.4 Navigable**
- [ ] Bypass blocks available
- [ ] Page titles identify topic
- [ ] Headings and labels describe topic or purpose
- [ ] Focus indication provided

**3.1 Readable**
- [ ] Language of page identified
- [ ] Language changes identified

**3.2 Predictable**
- [ ] Consistent navigation
- [ ] Consistent identification of components
- [ ] Changes of context only on user request

**3.3 Input Assistance**
- [ ] Error prevention for legal/financial data
- [ ] Labels or instructions provided
- [ ] Error suggestions provided
- [ ] Error identification provided

**4.1 Compatible**
- [ ] Parsing (no unclosed tags)
- [ ] Name, role, value available to assistive tech

## Testing Tools and Resources

### Browser Extensions

Use browser extensions for accessibility testing:

- **WAVE**: Evaluates web content for accessibility issues
- **axe DevTools**: Provides detailed accessibility reports
- **Lighthouse**: Includes accessibility audits
- **NoCoffee**: Simulates vision deficiencies

### Online Tools

- **WebAIM WAVE**: Free online accessibility evaluation tool
- **Accessibility Insights**: Microsoft's accessibility testing tool
- **Pa11y**: Automated accessibility testing tool

### Assistive Technology Testing

Test with actual assistive technologies:

- **Screen readers**: NVDA (Windows), VoiceOver (Mac), TalkBack (Android)
- **Keyboard navigation**: Navigate entirely without mouse
- **Screen magnifiers**: Zoom, MAGic, SuperNova
- **Voice recognition**: Dragon NaturallySpeaking, built-in voice control

## Summary

Creating an accessible Physical AI & Humanoid Robotics textbook requires attention to:

1. **Semantic HTML**: Use proper heading structure and semantic elements
2. **Alternative text**: Provide meaningful descriptions for all non-text content
3. **Keyboard navigation**: Ensure all functionality is accessible via keyboard
4. **Color and contrast**: Maintain sufficient color contrast ratios
5. **Forms and controls**: Properly label and associate form elements
6. **ARIA attributes**: Use appropriately when native semantics are insufficient
7. **Testing**: Regularly test with automated tools and assistive technologies
8. **Content**: Write in plain language with clear instructions

By following these guidelines, we ensure that the Physical AI & Humanoid Robotics textbook is accessible to all learners, regardless of their abilities or assistive technologies used.