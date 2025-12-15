/**
 * Chat utilities for session management
 * Implements the data models defined in specs/003-chatbot-ui/data-model.md
 */

// Define TypeScript interfaces
export interface IMessage {
  id: string;
  content: string;
  sender: 'user' | 'system';
  timestamp: Date;
  status: 'sent' | 'pending' | 'error';
}

export interface ISession {
  id: string;
  messages: IMessage[];
  createdAt: Date;
  updatedAt: Date;
  addMessage: (message: IMessage) => void;
  save: () => void;
  clear: () => void;
}

export interface IAPIResponse {
  status: string; // 'success' | 'error'
  query: string;
  result: string;
}

// ChatMessage class
export class ChatMessage implements IMessage {
  id: string;
  content: string;
  sender: 'user' | 'system';
  timestamp: Date;
  status: 'sent' | 'pending' | 'error';

  constructor(content: string, sender: 'user' | 'system', timestamp: Date = new Date()) {
    this.id = this.generateId();
    this.content = content;
    this.sender = sender;
    this.timestamp = timestamp;
    this.status = 'sent'; // Default status
  }

  private generateId(): string {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }
}

// ChatSession class
export class ChatSession implements ISession {
  id: string;
  messages: IMessage[];
  createdAt: Date;
  updatedAt: Date;

  constructor(id?: string) {
    this.id = id || this.generateId();
    this.messages = [];
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }

  private generateId(): string {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  addMessage(message: IMessage): void {
    this.messages.push(message);
    this.updatedAt = new Date();
  }

  // Save session to sessionStorage
  save(): void {
    try {
      sessionStorage.setItem(`chat_session_${this.id}`, JSON.stringify({
        id: this.id,
        messages: this.messages.map(msg => ({
          ...msg,
          timestamp: msg.timestamp.toISOString()
        })),
        createdAt: this.createdAt.toISOString(),
        updatedAt: this.updatedAt.toISOString()
      }));
    } catch (error) {
      console.error('Failed to save chat session:', error);
    }
  }

  // Load session from sessionStorage
  static load(sessionId: string): ChatSession | null {
    try {
      const sessionData = sessionStorage.getItem(`chat_session_${sessionId}`);
      if (sessionData) {
        const parsed = JSON.parse(sessionData);
        const session = new ChatSession(parsed.id);
        session.createdAt = new Date(parsed.createdAt);
        session.updatedAt = new Date(parsed.updatedAt);
        // Reconstruct messages
        session.messages = parsed.messages.map((msg: any) => {
          const message = new ChatMessage(msg.content, msg.sender as 'user' | 'system', new Date(msg.timestamp));
          message.id = msg.id;
          message.status = msg.status as 'sent' | 'pending' | 'error';
          return message;
        });
        return session;
      }
      return null;
    } catch (error) {
      console.error('Failed to load chat session:', error);
      return null;
    }
  }

  // Update session on page load to restore conversation
  static restoreSessionOnLoad(): ChatSession | null {
    const allSessionKeys = Object.keys(sessionStorage).filter(key =>
      key.startsWith('chat_session_')
    );

    // Return the most recently updated session
    if (allSessionKeys.length > 0) {
      const sessionsWithMetadata = allSessionKeys.map(key => {
        const sessionData = sessionStorage.getItem(key);
        if (sessionData) {
          const parsed = JSON.parse(sessionData);
          return {
            key,
            session: parsed,
            updatedAt: new Date(parsed.updatedAt)
          };
        }
        return null;
      }).filter(Boolean) as { key: string; session: any; updatedAt: Date }[];

      if (sessionsWithMetadata.length > 0) {
        sessionsWithMetadata.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
        const latestSession = sessionsWithMetadata[0];
        return ChatSession.load(latestSession.key.replace('chat_session_', ''));
      }
    }

    return null;
  }

  // Clear session from sessionStorage
  clear(): void {
    try {
      sessionStorage.removeItem(`chat_session_${this.id}`);
    } catch (error) {
      console.error('Failed to clear chat session:', error);
    }
  }
}

// APIResponse class
export class APIResponse implements IAPIResponse {
  status: string; // 'success' | 'error'
  query: string;
  result: string;

  constructor(status: string, query: string, result: string) {
    this.status = status;
    this.query = query;
    this.result = result;
  }
}

// Session management utilities
export const SessionManager = {
  // Create a new session
  createSession: (): ChatSession => {
    return new ChatSession();
  },

  // Get existing session or create new one
  getOrCreateSession: (sessionId: string | null = null): ChatSession => {
    if (sessionId) {
      const existing = ChatSession.load(sessionId);
      if (existing) {
        return existing;
      }
    }
    return new ChatSession();
  },

  // Get all available sessions
  getAllSessions: (): ChatSession[] => {
    const sessions: ChatSession[] = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('chat_session_')) {
        const sessionId = key.replace('chat_session_', '');
        const session = ChatSession.load(sessionId);
        if (session) {
          sessions.push(session);
        }
      }
    }
    return sessions;
  },

  // Clear all sessions
  clearAllSessions: (): void => {
    const keysToRemove: string[] = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('chat_session_')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => sessionStorage.removeItem(key));
  }
};

export default {
  ChatMessage,
  ChatSession,
  APIResponse,
  SessionManager
};