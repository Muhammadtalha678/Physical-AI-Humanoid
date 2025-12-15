import React from 'react';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

interface MessageRendererProps {
  content: string;
}

export const MessageRenderer: React.FC<MessageRendererProps> = ({ content }) => {
  // Configure marked to use HTML renderer
  marked.setOptions({
    breaks: true,
    gfm: true,
  });

  try {
    // Convert markdown to HTML
    const rawHtml = marked(content) as string;
    // Sanitize HTML to prevent XSS
    const sanitizedHtml = DOMPurify.sanitize(rawHtml);

    return <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />;
  } catch (error) {
    console.error('Error rendering markdown:', error);
    // Fallback to plain text if markdown rendering fails
    return <div>{content}</div>;
  }
};

export default MessageRenderer;