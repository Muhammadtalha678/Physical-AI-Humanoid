# ChatbotWidget Component

A React-based chatbot widget that integrates with the RAG backend API to provide documentation assistance to users.

## Features

- **Persistent Chat Sessions**: Maintains conversation history across page navigation
- **Markdown Support**: Renders API responses with proper markdown formatting
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Responsive Design**: Works on all device sizes
- **Accessibility**: Full keyboard navigation and screen reader support
- **Real-time Communication**: Instant communication with the RAG backend

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `initialOpenState` | boolean | `false` | Whether the chat window should be open by default |
| `apiEndpoint` | string | `config.apiEndpoint` | API endpoint for the backend service |

## Usage

The ChatbotWidget is automatically integrated into all Docusaurus pages via the Root theme component. No additional setup is required.

## Architecture

- `ChatbotWidget`: Main container component
- `ChatWindow`: Handles the chat interface
- `MessageList`: Displays conversation history
- `MessageInput`: Handles user input
- `MessageRenderer`: Renders markdown content from API responses
- `api-service.ts`: Handles communication with the backend API
- `chat-utils.ts`: Contains data models and session management utilities

## Environment Variables

The component supports the following environment variables:

- `REACT_APP_API_ENDPOINT` or `API_ENDPOINT`: Backend API endpoint URL
- `REACT_APP_API_TIMEOUT` or `API_TIMEOUT`: Request timeout in milliseconds
- `REACT_APP_MAX_RETRIES` or `MAX_RETRIES`: Maximum number of retry attempts
- `REACT_APP_RETRY_DELAY` or `RETRY_DELAY`: Delay between retries in milliseconds

## API Contract

The component follows the contract defined in `specs/003-chatbot-ui/contracts/api-contract.md`:

- POST to `/api/query` with `{ query: "user's question" }`
- Expects response in format: `{ status: "success", query: "original query", result: "formatted response" }`
- Handles error responses in format: `{ status: "error", query: "original query", result: "error message" }`