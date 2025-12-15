/**
 * API Service for Chatbot UI - Communicates with RAG backend
 * Implements the contract defined in specs/003-chatbot-ui/contracts/api-contract.md
 */

interface APIResponse {
  status: string; // 'success' | 'error'
  query: string;
  result: string;
}

interface QueryRequest {
  query: string;
}

import config from '../utils/config';

/**
 * Sends a query to the RAG backend API with enhanced error handling and timeout
 * @param {string} query - The user's question/query
 * @param {number} timeoutMs - Request timeout in milliseconds (default: from config)
 * @returns {Promise<APIResponse>} The API response object
 */
export const sendQueryToBackend = async (query: string, timeoutMs: number = config.timeoutMs): Promise<APIResponse> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(config.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: APIResponse = await response.json();

    // Validate response structure
    if (!data.status || !data.query || data.result === undefined) {
      throw new Error('Invalid response format from backend API');
    }

    return data;
  } catch (error: any) {
    clearTimeout(timeoutId);

    // Handle timeout specifically
    if (error.name === 'AbortError') {
      console.error('API call timed out:', error);
      return {
        status: 'error',
        query: query,
        result: 'Request timed out. Please try again.',
      };
    }

    // Handle network errors or other issues
    console.error('API call failed:', error);
    return {
      status: 'error',
      query: query,
      result: error.message || 'Network error occurred. Please check your connection and try again.',
    };
  }
};

/**
 * Sends a query to the RAG backend API with retry mechanism
 * @param {string} query - The user's question/query
 * @param {number} maxRetries - Maximum number of retry attempts (default: from config)
 * @param {number} retryDelayMs - Delay between retries in milliseconds (default: from config)
 * @returns {Promise<APIResponse>} The API response object
 */
export const sendQueryToBackendWithRetry = async (
  query: string,
  maxRetries: number = config.maxRetries,
  retryDelayMs: number = config.retryDelayMs
): Promise<APIResponse> => {
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const result = await sendQueryToBackend(query);

    // If successful or it's the last attempt, return the result
    if (result.status === 'success' || attempt === maxRetries) {
      return result;
    }

    // Store the error for potential return if all retries fail
    lastError = result;

    // Wait before retrying
    await new Promise(resolve => setTimeout(resolve, retryDelayMs * Math.pow(2, attempt))); // Exponential backoff
  }

  // This should not be reached, but just in case
  return lastError;
};

/**
 * Tests the connection to the backend API
 * @returns {Promise<boolean>} Whether the connection is successful
 */
export const testBackendConnection = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for connection test

    const response = await fetch(config.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: 'test connection' }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('Connection test failed:', error);
    return false;
  }
};

export default {
  sendQueryToBackend,
  sendQueryToBackendWithRetry,
  testBackendConnection,
};