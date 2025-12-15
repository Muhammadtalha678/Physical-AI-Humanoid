import React from 'react';
import { IMessage } from '../../utils/chat-utils';
import { MessageRenderer } from './MessageRenderer';
import './styles.css';

interface MessageListProps {
  messages: IMessage[];
  isLoading: boolean;
  error: string | null;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading, error }) => {
  return (
    <div className="message-list">
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {messages.map((message) => (
        <div
          key={message.id}
          className={`message ${message.sender === 'user' ? 'user-message' : 'system-message'}`}
        >
          <div className="message-content">
            {message.sender === 'user' ? message.content : <MessageRenderer content={message.content} />}
          </div>
          <div className="message-timestamp">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="loading-indicator">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;