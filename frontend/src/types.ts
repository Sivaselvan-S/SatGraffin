export type ChatRole = 'user' | 'assistant';

export interface ThinkingStep {
  step: string;
  message: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: string[];
  isError?: boolean;
  isAmbiguous?: boolean;
  disambiguationOptions?: string[];
  query?: string;
  thinkingSteps?: ThinkingStep[];
  imageUrl?: string;
  isStreaming?: boolean;
}

export interface QueryResponse {
  response: string;
  source_links?: string[];
  is_ambiguous?: boolean;
  disambiguation_options?: string[];
}
