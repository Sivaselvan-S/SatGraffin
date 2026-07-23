import { useCallback } from 'react';
import type { ThinkingStep } from '../types';


interface StreamOptions {
  apiBaseUrl: string;
  userId: string;
  model?: string;
  onThinkingStep?: (step: ThinkingStep) => void;
  onSources?: (sources: string[]) => void;
  onToken?: (token: string) => void;
  onError?: (error: string) => void;
  onDone?: () => void;
}

export function useStreamingQuery() {
  const streamQuery = useCallback(
    async (query: string, options: StreamOptions, file?: File) => {
      const { apiBaseUrl, userId, model, onThinkingStep, onSources, onToken, onError, onDone } = options;

      try {
        let uploadedImageUrl: string | undefined = undefined;

        // If a file is attached, upload it first via /api/upload
        if (file) {
          onThinkingStep?.({ step: 'upload', message: `Uploading ${file.name} for visual analysis...` });
          const formData = new FormData();
          formData.append('file', file);
          const uploadRes = await fetch(`${apiBaseUrl}/api/upload`, {
            method: 'POST',
            body: formData,
          });
          if (uploadRes.ok) {
            const uploadData = await uploadRes.json();
            uploadedImageUrl = uploadData.image_url || uploadData.file_url;
            onThinkingStep?.({ step: 'upload', message: `Uploaded visual: ${uploadData.filename || file.name}` });
          } else {
            console.warn('File upload failed, proceeding with text-only query');
          }
        }

        const response = await fetch(`${apiBaseUrl}/api/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query,
            user_id: userId,
            model,
            image_url: uploadedImageUrl,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        }

        if (!response.body) {
          throw new Error('ReadableStream not supported in this browser.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || ''; // keep incomplete line segment

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              const jsonStr = trimmed.slice(6);
              try {
                const event = JSON.parse(jsonStr);
                if (event.type === 'thinking') {
                  onThinkingStep?.({ step: event.step, message: event.message });
                } else if (event.type === 'sources') {
                  onSources?.(event.source_links || []);
                } else if (event.type === 'token') {
                  onToken?.(event.token || '');
                } else if (event.type === 'done') {
                  onDone?.();
                }
              } catch (e) {
                console.error('Error parsing SSE json:', e, jsonStr);
              }
            }
          }
        }

        onDone?.();
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Streaming request failed';
        onError?.(msg);
      }
    },
    []
  );

  return { streamQuery };
}
