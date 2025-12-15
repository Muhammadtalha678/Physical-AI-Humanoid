/**
 * Configuration utility for the Chatbot UI
 * Manages environment-specific settings
 */

// Define configuration interface
interface AppConfig {
  apiEndpoint: string;
  timeoutMs: number;
  maxRetries: number;
  retryDelayMs: number;
}

// Default configuration
const DEFAULT_CONFIG: AppConfig = {
  
  apiEndpoint: "https://rag-backend-22uh.onrender.com/api/query",
  timeoutMs: 10000, // 10 seconds
  maxRetries: 3,
  retryDelayMs: 1000, // 1 second
};

// Get configuration from environment variables or use defaults
// For Docusaurus, we'll use window-based configuration since process.env is not available in browser
const getConfig = (): AppConfig => {
  // In Docusaurus, environment variables are available through process.env during build time
  // but for runtime configuration, we can use window object or defaults
  
  if (typeof window !== 'undefined' && (window as any).CHATBOT_CONFIG) {
    // Configuration provided via window object
    const windowConfig = (window as any).CHATBOT_CONFIG;
    return {
      apiEndpoint: windowConfig.apiEndpoint || DEFAULT_CONFIG.apiEndpoint,
      timeoutMs: parseInt(windowConfig.timeoutMs || DEFAULT_CONFIG.timeoutMs.toString(), 10),
      maxRetries: parseInt(windowConfig.maxRetries || DEFAULT_CONFIG.maxRetries.toString(), 10),
      retryDelayMs: parseInt(windowConfig.retryDelayMs || DEFAULT_CONFIG.retryDelayMs.toString(), 10),
    };
  }

  // For Docusaurus, build-time environment variables would be processed during build
  // We'll use a simple approach with global variables that can be defined in docusaurus.config.js
  return {
    apiEndpoint: DEFAULT_CONFIG.apiEndpoint, // This can be overridden in docusaurus.config.js
    timeoutMs: DEFAULT_CONFIG.timeoutMs,
    maxRetries: DEFAULT_CONFIG.maxRetries,
    retryDelayMs: DEFAULT_CONFIG.retryDelayMs,
  };
};

// Validate configuration
const validateConfig = (config: AppConfig): boolean => {
  return (
    typeof config.apiEndpoint === 'string' &&
    config.apiEndpoint.length > 0 &&
    typeof config.timeoutMs === 'number' &&
    config.timeoutMs > 0 &&
    typeof config.maxRetries === 'number' &&
    config.maxRetries >= 0 &&
    typeof config.retryDelayMs === 'number' &&
    config.retryDelayMs > 0
  );
};

// Get validated configuration
const config: AppConfig = getConfig();

if (!validateConfig(config)) {
  console.warn('Invalid configuration detected, using defaults:', config);
  // Don't throw an error in the browser, just use defaults
}

export default config;