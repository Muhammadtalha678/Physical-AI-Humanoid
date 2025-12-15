# API Contract: Chatbot UI for Docusaurus Documentation

## Backend API Integration

### Query Endpoint
- **Method**: POST
- **URL**: https://rag-backend-22uh.onrender.com/api/query
- **Content-Type**: application/json

#### Request Body
```json
{
  "query": "string (the user's question)"
}
```

#### Success Response
```json
{
  "status": "success",
  "query": "string (echo of the original query)",
  "result": "string (formatted response with markdown support)"
}
```

#### Error Response
```json
{
  "status": "error",
  "query": "string (echo of the original query)",
  "result": "string (error message)"
}
```

### Frontend Integration Points

#### Chat Message Submission
- **Trigger**: User clicks send button or presses Enter
- **Action**: Send POST request to backend API
- **Payload**: User's query text
- **Expected Response**: Formatted result with markdown support

#### Message Display
- **Input**: API response object
- **Processing**: Render markdown content appropriately
- **Output**: Formatted message in chat interface

#### Error Handling
- **Trigger**: API returns error status or network failure
- **Action**: Display user-friendly error message
- **Options**: Retry mechanism available to user

## UI Component Contracts

### ChatWidget Component
- **Props**:
  - `initialOpenState`: boolean (whether to start open or minimized)
  - `apiEndpoint`: string (backend API URL)
- **Events**:
  - `onMessageSent`: triggered when user sends a message
  - `onMessageReceived`: triggered when response is received
  - `onError`: triggered when an error occurs

### Message Component
- **Props**:
  - `content`: string (message text/content)
  - `sender`: 'user' | 'system' (message origin)
  - `timestamp`: Date (when message was sent/received)
  - `status`: 'sent' | 'pending' | 'error' (delivery status)