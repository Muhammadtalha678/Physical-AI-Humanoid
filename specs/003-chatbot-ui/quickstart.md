# Quickstart: Chatbot UI for Docusaurus Documentation

## Prerequisites
- Node.js 18+ installed
- Docusaurus project set up
- Access to the backend API at https://rag-backend-22uh.onrender.com/api/query

## Installation Steps

1. **Install ChatKit dependency**
   ```bash
   npm install @pusher/chatkit-client
   # Or if using a different chat library as specified
   npm install chatkit
   ```

2. **Create the Chatbot component**
   ```bash
   # Create the component directory
   mkdir -p docs/src/components/ChatbotWidget
   touch docs/src/components/ChatbotWidget/index.js
   ```

3. **Add the global chatbot to Docusaurus**
   - Create/modify `docs/src/theme/Root.js` to include the chatbot widget
   - Add necessary CSS for positioning in bottom-right corner

4. **Configure API service**
   ```bash
   # Create API service for backend communication
   mkdir -p src/services
   touch src/services/api-service.js
   ```

## Basic Usage

1. **Run the Docusaurus development server**:
   ```bash
   cd docs
   npm start
   ```

2. **The chatbot widget will appear in the bottom-right corner** of every documentation page

3. **Test functionality**:
   - Click the chatbot icon to open the interface
   - Type a question and submit it
   - Verify that responses are displayed properly formatted

## Configuration

The chatbot can be configured through environment variables:
- `REACT_APP_CHATBOT_API_URL` - The backend API endpoint (defaults to https://rag-backend-22uh.onrender.com/api/query)

## Testing

1. **Unit tests**:
   ```bash
   npm run test:unit src/components/ChatbotWidget/
   ```

2. **Integration tests**:
   ```bash
   npm run test:integration src/services/api-service.js
   ```

3. **End-to-end tests**:
   ```bash
   npm run test:e2e
   ```