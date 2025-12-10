# Feature Specification: RAG Backend System

**Feature Branch**: `001-rag-backend`
**Created**: 2025-12-10
**Status**: Draft
**Input**: User description: "Create a Retrieval-Augmented Generation (RAG) backend system with fastapi, Qdrant vector db and OpenAIAgent sdk. Flow Two apis create using fast api one is for embeddings, this embedding api take the path of the directory where .md extension files are presented. These files are presented inside the docasarus project having directory frontend/docs/.md files,.md files,subfolders having .md files Second is for search query when user ask question the api hit of query this query than go to OpnAiAgentSDK than this agent see the query is related to greetings than the agentresponseke normal greeeting with "How can I help you with Physical AI & Humanoid Book",something like that in greeting, If the query is not related to greeting than call the agent tool with query than the tool search inside the qdrant vector db if the query related to book question than the agent  response with the same book response, if nothing found from book than the agent response like nothing found related to this query found, or if user ask any impropiate query than agent reply with "Be in a manner like message", This agents responses return back with the second api in response to the user. The qdrant_api ,qdrant_url,qdrant_ection_name,embedding_model,google_api_key,openrouteer_api,openrouter_baseurl,openrouter_model_name these all I provide."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.

  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Document Embedding Processing (Priority: P1)

As a system administrator, I want to upload or specify a directory containing .md files so that the system can process these documents and store their embeddings in the vector database for later retrieval.

**Why this priority**: This is foundational functionality - without processed documents, the search feature cannot work, making it the most critical component of the RAG system.

**Independent Test**: Can be fully tested by providing a directory path with .md files and verifying that embeddings are successfully created and stored in the Qdrant vector database.

**Acceptance Scenarios**:

1. **Given** a directory with .md files exists, **When** the embedding API is called with the directory path, **Then** all .md files (including those in subdirectories) are processed and their embeddings are stored in the vector database
2. **Given** a directory with .md files exists, **When** the embedding API processes nested subdirectories, **Then** all .md files regardless of their depth in the directory structure are included in the embedding process

---

### User Story 2 - Query Search and Response (Priority: P2)

As a user, I want to ask questions about the Physical AI & Humanoid Book content so that I can get relevant answers based on the documents processed by the system.

**Why this priority**: This is the primary user-facing functionality that delivers value by allowing users to interact with the processed document content.

**Independent Test**: Can be fully tested by submitting various types of queries (greetings, book-related questions, inappropriate queries) and verifying that appropriate responses are returned.

**Acceptance Scenarios**:

1. **Given** the system has processed document embeddings, **When** a user submits a greeting-type query, **Then** the system responds with "How can I help you with Physical AI & Humanoid Book" or similar greeting
2. **Given** the system has processed document embeddings, **When** a user submits a book-related question, **Then** the system retrieves relevant information from the vector database and provides an accurate response
3. **Given** the system has processed document embeddings, **When** a user submits a query with no relevant document matches, **Then** the system responds with "nothing found related to this query found"
4. **Given** the system is operational, **When** a user submits an inappropriate query, **Then** the system responds with "Be in a manner like message"

---

### User Story 3 - API Integration and Configuration (Priority: P3)

As a developer, I want to configure the system with external service credentials so that the RAG system can connect to Qdrant, OpenAI Agent SDK, and other services.

**Why this priority**: This enables the system to work with external services but is lower priority than the core functionality of processing documents and responding to queries.

**Independent Test**: Can be fully tested by configuring the system with various API keys and connection parameters and verifying connectivity to external services.

**Acceptance Scenarios**:

1. **Given** valid Qdrant API credentials, **When** the system attempts to connect to the Qdrant database, **Then** the connection is established successfully
2. **Given** valid OpenRouter API credentials, **When** the system attempts to connect to the OpenRouter service, **Then** the connection is established successfully

---

### Edge Cases

- What happens when the directory contains files that are not .md extensions?
- How does system handle extremely large .md files that might cause memory issues during processing?
- How does the system handle malformed or corrupted .md files during the embedding process?
- What happens when the Qdrant vector database is unavailable or returns errors?
- How does the system handle concurrent requests to the embedding API?
- What happens when a user submits a query that is extremely long or contains special characters?
- How does the system handle network timeouts when connecting to external services?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide an API endpoint to process .md files from a specified directory path and store their embeddings in a vector database
- **FR-002**: System MUST recursively scan subdirectories within the specified path to find all .md files for embedding processing
- **FR-003**: System MUST exclude non-.md files from the embedding processing workflow
- **FR-004**: System MUST provide an API endpoint for users to submit search queries about the Physical AI & Humanoid Book content
- **FR-005**: System MUST use OpenAI Agent SDK to process user queries and determine their intent
- **FR-006**: System MUST respond with a greeting message "How can I help you with Physical AI & Humanoid Book" when a query is identified as a greeting
- **FR-007**: System MUST search the Qdrant vector database for relevant document content when a query is not a greeting
- **FR-008**: System MUST return appropriate responses from the vector database when query-related content is found
- **FR-009**: System MUST return the message "nothing found related to this query found" when no relevant content is found in the vector database
- **FR-010**: System MUST return the message "Be in a manner like message" when an inappropriate query is detected
- **FR-011**: System MUST accept configuration parameters for Qdrant API (qdrant_api, qdrant_url, qdrant_collection_name)
- **FR-012**: System MUST accept configuration parameters for embedding model (embedding_model)
- **FR-013**: System MUST accept configuration parameters for OpenRouter service (openrouter_api, openrouter_baseurl, openrouter_model_name)
- **FR-014**: System MUST accept Google API key configuration parameter (google_api_key)
- **FR-015**: System MUST validate that the specified directory path exists and contains .md files before processing
- **FR-016**: System MUST handle errors gracefully when external services (Qdrant, OpenRouter) are unavailable
- **FR-017**: System MUST support processing of nested subdirectories containing .md files up to a reasonable depth limit

### Key Entities

- **Document Embedding**: Represents the vector representation of content from .md files, stored in the Qdrant vector database with associated metadata
- **Query Request**: Represents a user's input question that needs to be processed and matched against document embeddings
- **Response**: Represents the system's output to a user query, which may be a greeting, document-based answer, or appropriate error/inappropriate query response
- **Configuration Parameters**: Represents the external service credentials and settings needed for system operation (Qdrant, OpenRouter, embedding model, etc.)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System successfully processes 100% of .md files in a specified directory and its subdirectories within 5 minutes for directories containing up to 100 files
- **SC-002**: System responds to user queries within 10 seconds for 95% of requests under normal load conditions
- **SC-003**: 90% of relevant queries return accurate responses based on the document content when appropriate matches exist in the vector database
- **SC-004**: System correctly identifies and responds with appropriate greeting messages for 95% of greeting-type queries
- **SC-005**: System correctly identifies and responds with "Be in a manner like message" for 95% of inappropriate queries
- **SC-006**: System successfully handles 100 concurrent embedding processing requests without failure
- **SC-007**: 95% of queries with no relevant document matches return the "nothing found related to this query found" message
- **SC-008**: System maintains 99% uptime when external services (Qdrant, OpenRouter) are available
- **SC-009**: Users can successfully submit queries and receive responses 98% of the time during normal operation
- **SC-010**: System processes and stores embeddings for documents up to 10MB in size without memory errors
