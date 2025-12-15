import React, { useState } from 'react';
import './styles.css';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled: boolean;
}

export const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form className="message-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your question..."
        disabled={disabled}
        className="message-input"
      />
      <button
        type="submit"
        disabled={message.trim() === '' || disabled}
        className="send-button"
      >
        Send
      </button>
    </form>
  );
};

export default MessageInput;