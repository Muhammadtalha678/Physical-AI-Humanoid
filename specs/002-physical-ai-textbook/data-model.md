# Data Model: Physical AI & Humanoid Robotics Textbook

## Overview
This document defines the data models for the Physical AI & Humanoid Robotics textbook content management. The models represent the structure of educational content, assessments, and related metadata.

## Core Entities

### Textbook Chapter
Represents a week's worth of content with learning objectives, theory, and practical exercises.

**Fields:**
- `id` (string): Unique identifier for the chapter (e.g., "week-01-02-introduction")
- `title` (string): Descriptive title of the chapter
- `weekRange` (string): Week range covered (e.g., "1-2", "3-5", "11-12")
- `description` (string): Brief overview of the chapter content
- `learningObjectives` (array of strings): List of learning objectives
- `prerequisites` (array of strings): Prerequisite knowledge or chapters
- `contentSections` (array of objects): List of content sections within the chapter
- `assessments` (array of Assessment objects): Associated assessments
- `resources` (array of Resource objects): Downloadable resources
- `duration` (number): Estimated completion time in hours
- `difficulty` (string): Difficulty level (beginner, intermediate, advanced)

### Content Section
Represents individual sections within a chapter (e.g., lessons, topics).

**Fields:**
- `id` (string): Unique identifier for the section
- `title` (string): Section title
- `type` (string): Content type (lesson, exercise, demo, quiz, etc.)
- `content` (string): Markdown content for the section
- `order` (number): Display order within the chapter
- `interactiveElements` (array of objects): Interactive components (code playgrounds, simulators)

### Assessment Project
Hands-on project with specific deliverables and evaluation criteria.

**Fields:**
- `id` (string): Unique identifier for the assessment
- `title` (string): Assessment title
- `description` (string): Detailed description of the project
- `objectives` (array of strings): Learning objectives addressed
- `requirements` (array of strings): Technical requirements
- `deliverables` (array of strings): Expected deliverables
- `evaluationCriteria` (array of objects): Grading rubric
- `resources` (array of Resource objects): Required resources
- `estimatedTime` (number): Time needed to complete the project
- `complexity` (string): Complexity level (basic, intermediate, advanced)

### Resource
Downloadable or linked materials for the course.

**Fields:**
- `id` (string): Unique identifier for the resource
- `title` (string): Resource title
- `type` (string): Resource type (code-template, dataset, documentation, video, etc.)
- `url` (string): Path or URL to the resource
- `description` (string): Brief description of the resource
- `size` (string): File size if applicable
- `lastUpdated` (date): Last modification date

### Capstone Project
Final integration project combining all learned concepts.

**Fields:**
- `id` (string): Unique identifier for the capstone
- `title` (string): Capstone title
- `description` (string): Comprehensive description
- `requirements` (array of strings): All technical requirements
- `components` (array of strings): Required components (speech, planning, navigation, perception, manipulation)
- `platformSetups` (array of PlatformSetup objects): Supported platform configurations
- `deliverables` (array of strings): Final deliverables
- `evaluationCriteria` (array of objects): Comprehensive grading rubric
- `timeline` (object): Timeline with milestones

### PlatformSetup
Configuration options for different hardware/software setups.

**Fields:**
- `id` (string): Unique identifier for the setup
- `name` (string): Setup name (Digital Twin, Edge Kit, cloud-native)
- `requirements` (array of strings): Specific requirements for this setup
- `instructions` (string): Setup instructions in markdown
- `estimatedSetupTime` (number): Time to complete setup
- `costLevel` (string): Cost range (low, medium, high)

### UserProgress
Tracks learner progress through the course (if implemented with user accounts).

**Fields:**
- `userId` (string): Identifier for the user
- `chapterId` (string): Chapter being tracked
- `status` (string): Status (not-started, in-progress, completed)
- `completionDate` (date): When the chapter was completed
- `assessmentScores` (array of objects): Scores for assessments
- `timeSpent` (number): Time spent on the chapter in minutes

## Relationships

### Chapter Relationships
- One Chapter contains many Content Sections
- One Chapter has many Assessment Projects (optional)
- One Chapter has many Resources
- One Chapter may depend on other Chapters (prerequisites)

### Assessment Relationships
- One Assessment Project belongs to one Chapter
- One Assessment Project has many Resources
- One Assessment Project has many Evaluation Criteria

### Capstone Relationships
- One Capstone Project has many Platform Setups
- One Capstone Project has many Resources
- One Capstone Project has many Evaluation Criteria

### Resource Relationships
- One Resource can belong to many Chapters or Assessments

## Validation Rules

### Textbook Chapter
- `id` must follow the pattern "week-{number}-{number}-{description}"
- `weekRange` must be in the format "X-Y" where X and Y are numbers and X ≤ Y
- `learningObjectives` must contain at least one objective
- `duration` must be a positive number
- `difficulty` must be one of: "beginner", "intermediate", "advanced"

### Content Section
- `order` must be a positive integer
- `type` must be one of: "lesson", "exercise", "demo", "quiz", "assignment"
- `content` must be valid markdown

### Assessment Project
- `complexity` must be one of: "basic", "intermediate", "advanced"
- `estimatedTime` must be a positive number

### PlatformSetup
- `costLevel` must be one of: "low", "medium", "high"
- `name` must be one of: "Digital Twin", "Edge Kit", "cloud-native"

## State Transitions (if user tracking is implemented)

### Chapter Status
- `not-started` → `in-progress` (when user begins the chapter)
- `in-progress` → `completed` (when user completes all requirements)
- `completed` → `in-progress` (if user wants to review)

## Example Instances

### Example Chapter
```
{
  "id": "week-01-02-introduction",
  "title": "Introduction to Physical AI",
  "weekRange": "1-2",
  "description": "Foundations of Physical AI and embodied intelligence",
  "learningObjectives": [
    "Understand the difference between digital AI and physical AI",
    "Identify key components of humanoid robotics",
    "Explain sensor systems used in robotics"
  ],
  "prerequisites": [],
  "contentSections": [
    {
      "id": "foundations-physical-ai",
      "title": "Foundations of Physical AI",
      "type": "lesson",
      "content": "# Foundations of Physical AI\n\nPhysical AI represents...",
      "order": 1
    }
  ],
  "assessments": [],
  "resources": [],
  "duration": 8,
  "difficulty": "intermediate"
}
```

### Example Assessment
```
{
  "id": "ros2-package-project",
  "title": "ROS 2 Package Development Project",
  "description": "Develop a complete ROS 2 package with nodes, topics, and services",
  "objectives": [
    "Create ROS 2 nodes in Python",
    "Implement topic-based communication",
    "Design service-based interactions"
  ],
  "requirements": [
    "ROS 2 installation",
    "Python 3.8+",
    "Basic Python programming skills"
  ],
  "deliverables": [
    "Custom ROS 2 package with at least 2 nodes",
    "Documentation for the package",
    "Test results demonstrating functionality"
  ],
  "evaluationCriteria": [
    {
      "criterion": "Code Quality",
      "weight": 30,
      "description": "Well-structured, commented code following ROS 2 conventions"
    }
  ],
  "resources": [],
  "estimatedTime": 16,
  "complexity": "intermediate"
}
```

## Indexes and Performance Considerations

### Recommended Indexes
- Index on Chapter.id for quick lookups
- Index on Chapter.weekRange for range queries
- Index on UserProgress.userId for user-specific queries

### Performance Guidelines
- Limit content sections per chapter to 10-15 for optimal loading
- Optimize resource files for web delivery (compressed images, efficient formats)
- Cache static content appropriately