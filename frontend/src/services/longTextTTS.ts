import type {
  LongTextRequest,
  LongTextJobCreateResponse,
  LongTextJobResponse,
  LongTextJobList,
  LongTextSSEEvent,
  LongTextHistoryFilter,
  LongTextHistorySort,
  LongTextHistoryStats,
  LongTextJobDetails,
  LongTextJobUpdateRequest,
  LongTextJobRetryRequest,
  BulkJobActionRequest,
  BulkJobActionResponse,
  LongTextHistoryItem
} from '../types';
import { removeV1Suffix } from '../lib/utils';

export const createLongTextTTSService = (baseUrl: string, sessionId?: string) => {
  const service = {
    /**
     * Submit a new long text TTS job
     */
    submitJob: async (request: LongTextRequest): Promise<LongTextJobCreateResponse> => {
      // Build the JSON payload matching the backend LongTextRequest model
      const payload: any = {
        input: request.text, // Backend expects 'input' not 'text'
      };

      if (request.voice) {
        payload.voice = request.voice;
      }

      if (request.exaggeration !== undefined) {
        payload.exaggeration = request.exaggeration;
      }

      if (request.cfg_weight !== undefined) {
        payload.cfg_weight = request.cfg_weight;
      }

      if (request.temperature !== undefined) {
        payload.temperature = request.temperature;
      }

      if (request.output_format) {
        payload.response_format = request.output_format; // Backend expects 'response_format'
      }

      if (request.silence_padding_ms !== undefined) {
        payload.silence_padding_ms = request.silence_padding_ms;
      }

      if (request.chunking_strategy) {
        payload.chunking_strategy = request.chunking_strategy;
      }

      if (request.quality_preset) {
        payload.quality_preset = request.quality_preset;
      }

      if (request.chunk_size !== undefined) {
        payload.chunk_size = request.chunk_size;
      }

      // Add session ID for tracking
      if (sessionId) {
        payload.session_id = sessionId;
      }

      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Long text TTS job submission failed: ${response.status}`;

        try {
          const errorData = JSON.parse(errorText);
          if (errorData?.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          errorMessage += ` ${errorText}`;
        }

        throw new Error(errorMessage);
      }

      return response.json();
    },

    /**
     * Get job status and progress
     */
    getJobStatus: async (jobId: string): Promise<LongTextJobResponse> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get job status: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Download completed job audio
     */
    downloadJobAudio: async (jobId: string): Promise<Blob> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/download`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to download audio: ${response.status} ${errorText}`);
      }

      return response.blob();
    },

    /**
     * Pause a running job
     */
    pauseJob: async (jobId: string): Promise<{ message: string; status: string }> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/pause`, {
        method: 'PUT',
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to pause job: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Resume a paused job
     */
    resumeJob: async (jobId: string): Promise<{ message: string; status: string }> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/resume`, {
        method: 'PUT',
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to resume job: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Cancel/delete a job
     */
    cancelJob: async (jobId: string): Promise<{ message: string }> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to cancel job: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * List user's jobs with optional status filter
     */
    listJobs: async (options?: {
      status?: string;
      limit?: number;
      offset?: number;
    }): Promise<LongTextJobList> => {
      const params = new URLSearchParams();

      if (options?.status) {
        params.append('status', options.status);
      }

      if (options?.limit !== undefined) {
        params.append('limit', options.limit.toString());
      }

      if (options?.offset !== undefined) {
        params.append('offset', options.offset.toString());
      }

      // Session ID removed for filtering - show all jobs for better UX

      const url = `${removeV1Suffix(baseUrl)}/audio/speech/long${params.toString() ? `?${params.toString()}` : ''}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to list jobs: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Subscribe to Server-Sent Events for job progress
     */
    subscribeToSSE: (jobId: string, onEvent: (event: LongTextSSEEvent) => void, onError?: (error: Error) => void): (() => void) => {
      const eventSource = new EventSource(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/sse`);
      let hasReceivedCompletionEvent = false;

      // Handle named events
      eventSource.addEventListener('progress', (event) => {
        try {
          const data = JSON.parse(event.data);
          onEvent({
            job_id: jobId,
            timestamp: new Date().toISOString(),
            event_type: 'progress',
            data
          });
        } catch (error) {
          console.error('Failed to parse progress event:', error);
          onError?.(new Error('Failed to parse progress event'));
        }
      });

      eventSource.addEventListener('completed', (event) => {
        try {
          hasReceivedCompletionEvent = true;
          const data = JSON.parse(event.data);
          onEvent({
            job_id: jobId,
            timestamp: new Date().toISOString(),
            event_type: 'completed',
            data
          });
        } catch (error) {
          console.error('Failed to parse completed event:', error);
          onError?.(new Error('Failed to parse completed event'));
        }
      });

      eventSource.addEventListener('error', (event: MessageEvent) => {
        try {
          // Only parse if there's actual data to parse
          if (event.data && event.data.trim()) {
            const data = JSON.parse(event.data);
            onEvent({
              job_id: jobId,
              timestamp: new Date().toISOString(),
              event_type: 'error',
              data
            });
          }
          // If no data, this is likely a connection close event - ignore silently
        } catch (error) {
          // Only report parsing errors if we actually have data to parse
          if (event.data && event.data.trim()) {
            console.error('Failed to parse error event:', error);
            onError?.(new Error('Failed to parse error event'));
          }
        }
      });

      // Fallback for unnamed events
      eventSource.onmessage = (event) => {
        try {
          const data: LongTextSSEEvent = JSON.parse(event.data);
          onEvent({
            ...data,
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          console.error('Failed to parse SSE event:', error);
          onError?.(new Error('Failed to parse SSE event'));
        }
      };

      eventSource.onerror = (error) => {
        // Only report error if we haven't received a completion event
        // The EventSource API always fires error on connection close, even for intentional disconnection
        if (!hasReceivedCompletionEvent) {
          console.error('SSE connection error:', error);
          onError?.(new Error('SSE connection error'));
        } else {
          console.log('SSE connection closed after job completion');
        }
      };

      // Return cleanup function
      return () => {
        eventSource.close();
      };
    },

    /**
     * Estimate processing time for a given text length
     */
    estimateProcessingTime: (textLength: number): number => {
      // Rough estimate: 0.1 seconds per character
      // This matches the estimate from the implementation plan
      return Math.ceil(textLength * 0.1);
    },

    /**
     * Check if text should use long text processing
     */
    shouldUseLongText: (text: string): boolean => {
      return text.length > 3000;
    },

    /**
     * Get user-friendly status message
     */
    getStatusMessage: (status: string): string => {
      const statusMessages = {
        pending: 'Job queued for processing',
        chunking: 'Breaking text into chunks',
        processing: 'Generating audio',
        paused: 'Job paused',
        completed: 'Job completed successfully',
        failed: 'Job failed',
        cancelled: 'Job cancelled'
      };

      return statusMessages[status as keyof typeof statusMessages] || `Status: ${status}`;
    },

    // History Management API Methods

    /**
     * List history jobs with advanced filtering and sorting
     */
    listHistoryJobs: async (options?: {
      filters?: LongTextHistoryFilter;
      sort?: LongTextHistorySort;
      limit?: number;
      offset?: number;
    }): Promise<{ jobs: LongTextHistoryItem[]; total_count: number }> => {
      const params = new URLSearchParams();

      if (options?.filters) {
        Object.entries(options.filters).forEach(([key, value]) => {
          if (value !== undefined && value !== null && value !== '') {
            // Map frontend filter names to backend API parameter names
            const backendKey = key === 'search_text' ? 'search' : key;
            params.append(backendKey, value.toString());
          }
        });
      }

      if (options?.sort) {
        params.append('sort', options.sort);
      }

      if (options?.limit !== undefined) {
        params.append('limit', options.limit.toString());
      }

      if (options?.offset !== undefined) {
        params.append('offset', options.offset.toString());
      }

      // Session ID removed for filtering - show all jobs for better UX

      const url = `${removeV1Suffix(baseUrl)}/audio/speech/long-history${params.toString() ? `?${params.toString()}` : ''}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to list history jobs: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      // Transform backend response: map total_jobs to total_count for frontend compatibility
      return {
        jobs: data.jobs,
        total_count: data.total_jobs
      };
    },

    /**
     * Get detailed statistics about user's history
     */
    getHistoryStats: async (): Promise<LongTextHistoryStats> => {
      const params = new URLSearchParams();

      // Session ID removed for filtering - show all jobs for better UX

      const url = `${removeV1Suffix(baseUrl)}/audio/speech/long-history/stats${params.toString() ? `?${params.toString()}` : ''}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get history stats: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Get detailed information about a specific job
     */
    getJobDetails: async (jobId: string): Promise<LongTextJobDetails> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/details`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to get job details: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Update job metadata (name, tags, archived status)
     */
    updateJob: async (jobId: string, updates: LongTextJobUpdateRequest): Promise<{ message: string }> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update job: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Retry a failed job with optional new parameters
     */
    retryJob: async (jobId: string, retryOptions?: LongTextJobRetryRequest): Promise<LongTextJobCreateResponse> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/${jobId}/retry`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(retryOptions || {}),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to retry job: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Clear all history (with confirmation)
     */
    clearHistory: async (confirm: boolean = false): Promise<{ message: string; deleted_count: number }> => {
      const params = new URLSearchParams();
      params.append('confirm', confirm.toString());

      // Session ID removed for filtering - show all jobs for better UX

      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long-history?${params.toString()}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to clear history: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Perform bulk actions on multiple jobs
     */
    bulkAction: async (request: BulkJobActionRequest): Promise<BulkJobActionResponse> => {
      const response = await fetch(`${removeV1Suffix(baseUrl)}/audio/speech/long/bulk`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to perform bulk action: ${response.status} ${errorText}`);
      }

      return response.json();
    },

    /**
     * Archive a job (move to archived status)
     */
    archiveJob: async (jobId: string): Promise<{ message: string }> => {
      return service.updateJob(jobId, { is_archived: true });
    },

    /**
     * Unarchive a job (restore from archived status)
     */
    unarchiveJob: async (jobId: string): Promise<{ message: string }> => {
      return service.updateJob(jobId, { is_archived: false });
    },

    /**
     * Search jobs by text content
     */
    searchJobs: async (searchText: string, options?: {
      limit?: number;
      offset?: number;
    }): Promise<{ jobs: LongTextHistoryItem[]; total_count: number }> => {
      return service.listHistoryJobs({
        filters: { search_text: searchText },
        ...options
      });
    }
  };

  return service;
};

export type LongTextTTSService = ReturnType<typeof createLongTextTTSService>;