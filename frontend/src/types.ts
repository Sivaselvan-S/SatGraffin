export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: string[];
  isError?: boolean;
}

export interface QueryResponse {
  response: string;
  source_links?: string[];
}
