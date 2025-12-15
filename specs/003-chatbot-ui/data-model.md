# Data Model: Chatbot UI for Docusaurus Documentation

## Entities

### ChatMessage
- **id**: string (unique identifier for the message)
- **content**: string (the actual text content of the message)
- **sender**: 'user' | 'system' (indicates whether the message was sent by user or received from system)
- **timestamp**: Date (when the message was created/sent)
- **status**: 'sent' | 'pending' | 'error' (for tracking message delivery status)

### ChatSession
- **id**: string (unique identifier for the session)
- **messages**: ChatMessage[] (list of messages in the conversation)
- **createdAt**: Date (when the session was started)
- **updatedAt**: Date (when the session was last updated)

### APIResponse
- **status**: string (API response status - 'success' or 'error')
- **query**: string (the original query sent to the backend)
- **result**: string (the formatted response from the backend API)

## State Management

### ChatWindowState
- **isOpen**: boolean (whether the chat window is currently open or minimized)
- **isLoading**: boolean (whether an API request is currently in progress)
- **currentInput**: string (the current text in the input field)
- **sessionId**: string | null (current session ID if exists)
- **error**: string | null (any error messages to display)

## Validation Rules

### From Functional Requirements:
- **FR-001**: Chat interface must be accessible from all documentation pages
- **FR-002**: Must use ChatKit library for UI implementation
- **FR-003**: Must send user queries to the specified backend API endpoint
- **FR-004**: Must display messages in conversational format
- **FR-005**: Must properly format markdown content from API responses
- **FR-006**: Must handle API errors gracefully
- **FR-007**: Must preserve conversation state across page navigation
- **FR-008**: Must provide clear close/minimize functionality
- **FR-009**: Must not interfere with main documentation content