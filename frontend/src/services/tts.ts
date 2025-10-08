import axios from 'axios';
import type { TTSRequest, HealthResponse, VoiceLibraryResponse, DefaultVoiceResponse, VoiceLibraryItem, SSEEvent, StreamingProgress, SupportedLanguagesResponse } from '../types';

export const createTTSService = (baseUrl: string, sessionId?: string) => ({
  generateSpeech: async (request: TTSRequest): Promise<Blob> => {
    const formData = new FormData();
    formData.append('input', request.input);

    if (request.voice) {
      formData.append('voice', request.voice);
    }

    if (request.exaggeration !== undefined) {
      formData.append('exaggeration', request.exaggeration.toString());
    }

    if (request.cfg_weight !== undefined) {
      formData.append('cfg_weight', request.cfg_weight.toString());
    }

    if (request.temperature !== undefined) {
      formData.append('temperature', request.temperature.toString());
    }

    if (request.voice_file) {
      formData.append('voice_file', request.voice_file);
    }

    // Add session ID for tracking
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await fetch(`${baseUrl}/audio/speech/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`TTS generation failed: ${response.status} ${errorText}`);
    }

    return response.blob();
  },

  generateSpeechJSON: async (request: TTSRequest): Promise<Blob> => {
    const response = await axios.post(`${baseUrl}/audio/speech`, request, {
      responseType: 'blob',
      headers: { 'Content-Type': 'application/json' }
    });
    return response.data;
  },

  // SSE Streaming method
  generateSpeechSSE: async function* (request: TTSRequest): AsyncGenerator<{ event: SSEEvent; progress: StreamingProgress }> {
    const requestData = {
      ...request,
      stream_format: 'sse' as const
    };

    const response = await fetch(`${baseUrl}/audio/speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`SSE streaming failed: ${response.status} ${errorText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const progress: StreamingProgress = {
      chunksReceived: 0,
      totalBytes: 0,
      isComplete: false,
      audioChunks: []
    };

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          progress.isComplete = true;
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const eventData = line.slice(6);

            try {
              const event: SSEEvent = JSON.parse(eventData);

              if (event.type === 'speech.audio.delta') {
                // For SSE, we don't create blobs here, just count bytes and chunks.
                // The hook will handle decoding and processing.
                const audioData = atob(event.audio);
                progress.chunksReceived++;
                progress.totalBytes += audioData.length;
              }

              yield { event, progress: { ...progress } };

              if (event.type === 'speech.audio.done') {
                progress.isComplete = true;
                return;
              }
            } catch (e) {
              console.warn('Failed to parse SSE event:', eventData);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  // Raw audio streaming method
  generateSpeechStream: async function* (request: TTSRequest): AsyncGenerator<{ chunk: Blob; progress: StreamingProgress }> {
    const formData = new FormData();
    formData.append('input', request.input);

    if (request.voice) {
      formData.append('voice', request.voice);
    }

    if (request.exaggeration !== undefined) {
      formData.append('exaggeration', request.exaggeration.toString());
    }

    if (request.cfg_weight !== undefined) {
      formData.append('cfg_weight', request.cfg_weight.toString());
    }

    if (request.temperature !== undefined) {
      formData.append('temperature', request.temperature.toString());
    }

    if (request.voice_file) {
      formData.append('voice_file', request.voice_file);
    }

    if (request.streaming_chunk_size !== undefined) {
      formData.append('streaming_chunk_size', request.streaming_chunk_size.toString());
    }

    if (request.streaming_strategy) {
      formData.append('streaming_strategy', request.streaming_strategy);
    }

    if (request.streaming_quality) {
      formData.append('streaming_quality', request.streaming_quality);
    }

    // Add session ID for tracking
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await fetch(`${baseUrl}/audio/speech/stream/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Audio streaming failed: ${response.status} ${errorText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();

    const progress: StreamingProgress = {
      chunksReceived: 0,
      totalBytes: 0,
      isComplete: false,
      audioChunks: []
    };

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          progress.isComplete = true;
          break;
        }

        const chunk = new Blob([value], { type: 'audio/wav' });
        progress.audioChunks.push(chunk);
        progress.chunksReceived++;
        progress.totalBytes += chunk.size;

        yield { chunk, progress: { ...progress } };
      }
    } finally {
      reader.releaseLock();
    }
  },

  getHealth: async (): Promise<HealthResponse> => {
    const response = await fetch(`${baseUrl}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  },

  ping: async (): Promise<{ status: string; message: string }> => {
    const response = await fetch(`${baseUrl}/ping`);
    if (!response.ok) {
      throw new Error(`Ping failed: ${response.status}`);
    }
    return response.json();
  },

  // Voice library methods
  getVoices: async (): Promise<VoiceLibraryResponse> => {
    const response = await fetch(`${baseUrl}/audio/voices`);
    if (!response.ok) {
      throw new Error(`Failed to fetch voices: ${response.status}`);
    }
    return response.json();
  },

  getSupportedLanguages: async (): Promise<SupportedLanguagesResponse> => {
    const response = await fetch(`${baseUrl}/languages`);
    if (!response.ok) {
      throw new Error(`Failed to fetch supported languages: ${response.status}`);
    }
    return response.json();
  },

  uploadVoice: async (voiceName: string, voiceFile: File, language?: string): Promise<any> => {
    const formData = new FormData();
    formData.append('voice_name', voiceName);
    formData.append('voice_file', voiceFile);
    if (language) {
      formData.append('language', language);
    }

    const response = await fetch(`${baseUrl}/voices`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Upload failed: ${response.status}`);
    }

    return response.json();
  },

  deleteVoice: async (voiceName: string): Promise<void> => {
    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Delete failed: ${response.status}`);
    }
  },

  renameVoice: async (voiceName: string, newName: string): Promise<void> => {
    const formData = new FormData();
    formData.append('new_name', newName);

    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}`, {
      method: 'PUT',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Rename failed: ${response.status}`);
    }
  },

  downloadVoice: async (voiceName: string): Promise<Blob> => {
    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}/download`);
    if (!response.ok) {
      throw new Error(`Failed to download voice: ${response.status}`);
    }
    return response.blob();
  },

  getVoiceInfo: async (voiceName: string): Promise<VoiceLibraryItem> => {
    const response = await axios.get(`${baseUrl}/voices/${encodeURIComponent(voiceName)}`);
    return response.data.voice;
  },

  // New status endpoints
  getStatus: async (options?: {
    includeMemory?: boolean;
    includeHistory?: boolean;
    includeStats?: boolean;
    historyLimit?: number;
  }) => {
    const params = new URLSearchParams();
    if (options?.includeMemory) params.append('include_memory', 'true');
    if (options?.includeHistory) params.append('include_history', 'true');
    if (options?.includeStats) params.append('include_stats', 'true');
    if (options?.historyLimit) params.append('history_limit', options.historyLimit.toString());

    const response = await axios.get(`${baseUrl}/status?${params.toString()}`);
    return response.data;
  },

  getProgress: async () => {
    const response = await axios.get(`${baseUrl}/status/progress`);
    return response.data;
  },

  getStatistics: async (includeMemory = false) => {
    const params = includeMemory ? '?include_memory=true' : '';
    const response = await axios.get(`${baseUrl}/status/statistics${params}`);
    return response.data;
  },

  getMemoryInfo: async () => {
    const response = await axios.get(`${baseUrl}/memory`);
    return response.data;
  },

  clearMemory: async () => {
    const response = await axios.post(`${baseUrl}/memory/reset`);
    return response.data;
  },

  getApiInfo: async () => {
    const response = await axios.get(`${baseUrl}/info`);
    return response.data;
  },

  // Default voice management
  getDefaultVoice: async (): Promise<DefaultVoiceResponse> => {
    const response = await fetch(`${baseUrl}/voices/default`);
    if (!response.ok) {
      throw new Error(`Failed to get default voice: ${response.status}`);
    }
    return response.json();
  },

  setDefaultVoice: async (voiceName: string): Promise<any> => {
    const formData = new FormData();
    formData.append('voice_name', voiceName);

    const response = await fetch(`${baseUrl}/voices/default`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to set default voice: ${response.status}`);
    }

    return response.json();
  },

  clearDefaultVoice: async (): Promise<any> => {
    const response = await fetch(`${baseUrl}/voices/default`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to clear default voice: ${response.status}`);
    }

    return response.json();
  },

  // Alias management methods
  addAlias: async (voiceName: string, alias: string): Promise<any> => {
    const formData = new FormData();
    formData.append('alias', alias);

    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}/aliases`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to add alias: ${response.status}`);
    }

    return response.json();
  },

  removeAlias: async (voiceName: string, alias: string): Promise<any> => {
    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}/aliases/${encodeURIComponent(alias)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to remove alias: ${response.status}`);
    }

    return response.json();
  },

  listAliases: async (voiceName: string): Promise<{ voice_name: string; aliases: string[]; count: number }> => {
    const response = await fetch(`${baseUrl}/voices/${encodeURIComponent(voiceName)}/aliases`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to list aliases: ${response.status}`);
    }

    return response.json();
  },

  getAllVoiceNames: async (): Promise<{ voice_names: string[]; count: number }> => {
    const response = await fetch(`${baseUrl}/voices/all-names`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData?.error?.message || `Failed to get all voice names: ${response.status}`);
    }

    return response.json();
  },

  // Advanced memory management endpoints
  getMemoryConfig: async () => {
    const response = await axios.get(`${baseUrl}/memory/config`);
    return response.data;
  },

  updateMemoryConfig: async (config: { cpu_memory_percent?: number; gpu_memory_mb?: number }) => {
    const params = new URLSearchParams();
    if (config.cpu_memory_percent !== undefined) {
      params.append('cpu_memory_percent', config.cpu_memory_percent.toString());
    }
    if (config.gpu_memory_mb !== undefined) {
      params.append('gpu_memory_mb', config.gpu_memory_mb.toString());
    }

    const response = await axios.post(`${baseUrl}/memory/config?${params.toString()}`);
    return response.data;
  },

  getMemoryRecommendations: async () => {
    const response = await axios.get(`${baseUrl}/memory/recommendations`);
    return response.data;
  },

  getMemoryWithAlerts: async (includeAlerts = true) => {
    const params = includeAlerts ? '?include_alerts=true' : '';
    const response = await axios.get(`${baseUrl}/memory${params}`);
    return response.data;
  },
});

export type TTSService = ReturnType<typeof createTTSService>; 