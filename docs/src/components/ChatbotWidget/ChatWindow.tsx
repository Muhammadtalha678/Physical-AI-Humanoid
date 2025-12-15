import React, { useState, useEffect } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { sendQueryToBackend, sendQueryToBackendWithRetry } from '../../services/api-service';
import { ChatSession, SessionManager, ChatMessage, IMessage } from '../../utils/chat-utils';
import config from '../../utils/config';
import './styles.css';

interface ChatWindowProps {
  sessionId: string | null;
  apiEndpoint: string;
  onClose: () => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ sessionId, apiEndpoint, onClose }) => {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize session and messages
  useEffect(() => {
    if (sessionId) {
      const chatSession = SessionManager.getOrCreateSession(sessionId);
      setSession(chatSession);
      setMessages(chatSession.messages);
    }
  }, [sessionId]);

  const handleSendMessage = async (messageText: string) => {
    if (!session || !messageText.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to session
      const userMessage = new ChatMessage(messageText, 'user');
      session.addMessage(userMessage);

      // Update local messages state immediately
      setMessages([...session.messages]);

      // Send to backend with retry mechanism and timeout
      const response = await sendQueryToBackendWithRetry(messageText, 3, 1000);

      // Add system response to session
      const systemMessage = new ChatMessage(response.result, 'system');
      session.addMessage(systemMessage);

      // Save session
      session.save();

      // Update local messages state
      setMessages([...session.messages]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h3>Documentation Assistant</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      {session ? (
        <>
          <MessageList
            messages={messages}
            isLoading={isLoading}
            error={error}
          />
          <MessageInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
          />
        </>
      ) : (
        <div className="loading">Loading chat...</div>
      )}
    </div>
  );
};

export default ChatWindow;