---
id: 1
title: Create Constitution
stage: constitution
date: 2025-12-08
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user:
command: /sp.constitution
labels: [constitution,physical-ai,humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
Project: Physical AI & Humanoid Robotics Course
Requirements
You are required to complete a unified book project using Claude Code and Spec-Kit Plus. The core deliverables are:

1. AI/Spec-Driven Book Creation: Write a book using Docusaurus and deploy it to GitHub Pages. You will use Spec-Kit Plus ( https://github.com/panaversity/spec-kit-plus/ ) and Claude Code ( https://www.claude.com/product/claude-code ) to write the book.

2. Integrated RAG Chatbot Development: Build and embed a Retrieval-Augmented Generation (RAG) chatbot within the published book. This chatbot, utilizing the OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres database, and Qdrant Cloud Free Tier, must be able to answer user questions about the book's content, including answering questions based only on text selected by the user.

3. creating and using reusable intelligence via Claude Code Subagents and Agent Skills in the book project.

4. implement Signup and Signin using https://www.better-auth.com/ At signup you will ask questions from the user about their software and hardware background. Knowing the background of the user we will be able to personalize the content.
6.  the logged user can personalise the content in the chapters by pressing a button at the start of each chapter.

7.  the logged user can translate the content in Urdu in the chapters by pressing a button at the start of each chapter.
Here is the refrence from how to write constitution

Project: Research paper on AI-native software development

Core principles:
- Accuracy through primary source verification
- Clarity for academic audience (computer science background)
- Reproducibility (all claims cited and traceable)
- Rigor (peer-reviewed sources preferred)

Key standards:
- All factual claims must be traceable to sources
- Citation format: APA style
- Source types: minimum 50% peer-reviewed articles
- Plagiarism check: 0% tolerance before submission
- Writing clarity: Flesch-Kincaid grade 10-12

Constraints:
- Word count: 5,000-7,000 words
- Minimum 15 sources
- Format: PDF with embedded citations

Success criteria:
- All claims verified against sources
- Zero plagiarism detected
- Passes fact-checking review
```

## Response snapshot

Created a comprehensive constitution for the Physical AI & Humanoid Robotics Course project that addresses all specified deliverables. The constitution includes six core principles:

1. AI-Driven Development & Spec-First Approach
2. Educational Excellence & Accessibility
3. Test-First Development (NON-NEGOTIABLE)
4. Safe Physical AI Integration
5. Modular Architecture & Reusability
6. Real-Time Interaction & Responsiveness

Additional sections cover technology stack requirements, security & privacy standards, performance standards, development workflow, and governance policies. The constitution addresses all project requirements including Docusaurus book creation, RAG chatbot development, authentication with Better-Auth, personalization features, and Urdu translation capabilities.

## Outcome

- ✅ Impact: Created initial project constitution with comprehensive principles addressing all deliverables
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with complete constitution
- 🔁 Next prompts: Plan development tasks using /sp.plan
- 🧠 Reflection: Constitution provides solid foundation for project governance and development standards

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin spec creation using /sp.specify