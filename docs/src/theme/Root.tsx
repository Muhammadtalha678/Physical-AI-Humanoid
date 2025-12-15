import React from 'react';
import ChatbotWidget from '../components/ChatbotWidget';

// Default wrapper for the entire Docusaurus application
const Root = ({ children }: { children: React.ReactNode }) => {
  return (
    <>
      {children}
      <ChatbotWidget />
    </>
  );
};

export default Root;