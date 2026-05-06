export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: string[];
  isError?: boolean;
  isAmbiguous?: boolean;
  disambiguationOptions?: string[];
}

export interface QueryResponse {
  response: string;
  source_links?: string[];
  is_ambiguous?: boolean;
  disambiguation_options?: string[];
}
