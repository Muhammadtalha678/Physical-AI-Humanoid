import React, { useState, useEffect } from 'react';
import { ChatWindow } from './ChatWindow';
import { ChatSession } from '../../utils/chat-utils';
import config from '../../utils/config';
import './styles.css';

interface ChatbotWidgetProps {
  initialOpenState?: boolean;
  apiEndpoint?: string;
}

export const ChatbotWidget: React.FC<ChatbotWidgetProps> = ({
  initialOpenState = false,
  apiEndpoint = 'https://rag-backend-22uh.onrender.com/api/query'
}) => {
  const [isOpen, setIsOpen] = useState(initialOpenState);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Initialize session on component mount
  useEffect(() => {
    // Try to restore existing session or create new one
    const restoredSession = ChatSession.restoreSessionOnLoad();

    if (restoredSession) {
      // Use the restored session
      setSessionId(restoredSession.id);
    } else {
      // Create new session
      const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      setSessionId(newSessionId);
    }
  }, []);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <ChatWindow
          sessionId={sessionId}
          apiEndpoint={apiEndpoint}
          onClose={() => setIsOpen(false)}
        />
      ) : (
        <button className="chatbot-toggle-button" onClick={toggleChat}>
          <span className="chatbot-icon">💬</span>
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;