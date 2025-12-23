export interface TTSRequest {
  input: string;
  voice?: string;
  exaggeration?: number;
  cfg_weight?: number;
  temperature?: number;
  voice_file?: File;
  session_id?: string;
  stream_format?: 'audio' | 'sse';
  streaming_chunk_size?: number;
  streaming_strategy?: 'sentence' | 'paragraph' | 'fixed' | 'word';
  streaming_quality?: 'fast' | 'balanced' | 'high';
  enable_pauses?: boolean;
  custom_pauses?: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  config: any;
  memory_info?: {
    cpu_memory_mb: number;
    gpu_memory_allocated_mb: number;
  };
  initialization_state?: string;
  initialization_progress?: string;
  initialization_error?: string;
}

// New status API types
export interface TTSProgress {
  is_processing: boolean;
  status: string;
  current_step?: string;
  current_chunk?: number;
  total_chunks?: number;
  progress_percentage?: number;
  duration_seconds?: number;
  estimated_completion?: number;
  text_preview?: string;
  message?: string;
  request_id?: string;
  session_id?: string;
}

export interface TTSStatistics {
  total_requests: number;
  completed_requests: number;
  error_requests: number;
  success_rate: number;
  average_duration_seconds: number;
  average_text_length: number;
  is_processing: boolean;
  current_memory?: {
    cpu_memory_mb: number;
    gpu_memory_allocated_mb: number;
  };
}

export interface TTSRequestHistory {
  request_id: string;
  status: string;
  start_time: number;
  end_time?: number;
  duration_seconds: number;
  text_length: number;
  text_preview: string;
  voice_source: string;
  parameters: {
    exaggeration: number;
    cfg_weight: number;
    temperature: number;
  };
}

export interface TTSStatus {
  status: string;
  is_processing: boolean;
  request_id?: string;
  start_time?: number;
  duration_seconds?: number;
  text_length?: number;
  text_preview?: string;
  voice_source?: string;
  parameters?: {
    exaggeration: number;
    cfg_weight: number;
    temperature: number;
  };
  progress?: {
    current_chunk: number;
    total_chunks: number;
    current_step: string;
    progress_percentage: number;
    estimated_completion?: number;
  };
  total_requests: number;
  message?: string;
}

export interface APIInfo {
  api_name: string;
  version: string;
  status: string;
  tts_status: TTSStatus;
  statistics: TTSStatistics;
  memory_info?: {
    cpu_memory_mb: number;
    gpu_memory_allocated_mb: number;
  };
  recent_requests: TTSRequestHistory[];
  uptime_info: {
    total_requests: number;
    success_rate: number;
    is_processing: boolean;
  };
}

export interface VoiceSample {
  id: string;
  name: string;
  file: File;
  audioUrl: string;
  uploadDate: Date;
  aliases?: string[];
  language?: string;
}

export interface AudioRecord {
  id: string;
  name: string;
  audioUrl: string;
  blob: Blob;
  createdAt: Date;
  // duration?: number; // Duration in seconds
  duration?: number | null | undefined;
  settings: {
    text: string;
    exaggeration: number;
    cfgWeight: number;
    temperature: number;
    voiceId?: string;
    voiceName?: string;
  };
}

export interface VoiceLibraryItem {
  name: string;
  filename: string;
  original_filename: string;
  file_extension: string;
  file_size: number;
  upload_date: string;
  path: string;
  aliases?: string[];
  language?: string;
}

export interface VoiceLibraryResponse {
  voices: VoiceLibraryItem[];
  count: number;
}

export interface DefaultVoiceResponse {
  default_voice: string | null;
  source: 'voice_library' | 'file_system';
  voice_info?: VoiceLibraryItem;
  path?: string;
}

export interface SupportedLanguageItem {
  code: string;
  name: string;
}

export interface SupportedLanguagesResponse {
  languages: SupportedLanguageItem[];
  count: number;
}

export interface AudioInfo {
  sample_rate: number;
  channels: number;
  bits_per_sample: number;
}

export interface SSEAudioInfo {
  type: 'speech.audio.info';
  sample_rate: number;
  channels: number;
  bits_per_sample: number;
}

export interface SSEAudioDelta {
  type: 'speech.audio.delta';
  audio: string; // Base64 encoded audio chunk
}

export interface SSEAudioDone {
  type: 'speech.audio.done';
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

export type SSEEvent = SSEAudioInfo | SSEAudioDelta | SSEAudioDone;

export interface StreamingProgress {
  chunksReceived: number;
  totalBytes: number;
  isComplete: boolean;
  audioChunks: Blob[];
  currentChunk?: number;
  totalChunks?: number;
}

// Long Text TTS Types
export type LongTextJobStatus = 'pending' | 'chunking' | 'processing' | 'paused' | 'completed' | 'failed' | 'cancelled' | string;

export interface LongTextChunk {
  index: number;
  text: string;
  start_pos: number;
  end_pos: number;
  char_count: number;
  audio_file_path?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | string;
  processing_time_ms?: number;
  error?: string;
}

export interface LongTextProgress {
  job_id: string;
  overall_progress: number;
  current_chunk?: {
    index: number;
    text_preview: string;
    progress: number;
    started_at: string;
  };
  chunks_completed: Array<{
    index: number;
    duration_ms: number;
    audio_file: string;
  }>;
  estimated_remaining_seconds?: number;
}

export interface LongTextJobMetadata {
  job_id: string;
  created_at: string;
  updated_at: string;
  status: LongTextJobStatus;
  text_length: number;
  text_hash: string;
  chunks: {
    total: number;
    completed: number;
    current?: number;
    failed: number[];
  };
  voice: string;
  parameters: {
    exaggeration?: number;
    cfg_weight?: number;
    temperature?: number;
    output_format?: string;
    chunking_strategy?: 'sentence' | 'paragraph' | 'word' | 'fixed';
    chunk_size?: number;
    quality_preset?: 'fast' | 'balanced' | 'high';
    silence_padding_ms?: number;
    language?: string;
    [key: string]: unknown;
  };
  processing: {
    started_at?: string;
    paused_at?: string;
    completed_at?: string;
    total_processing_time_ms: number;
  };
  output: {
    format: string;
    path?: string;
    size_bytes?: number;
    duration_seconds?: number;
  };
  error?: string;
  user_session_id?: string;
}

export interface LongTextJobResponse {
  job: LongTextJobMetadata;
  progress?: LongTextProgress;
}

export interface LongTextJobListItem {
  job_id: string;
  status: LongTextJobStatus;
  created_at: string;
  updated_at: string;
  text_length: number;
  text_preview: string;
  voice: string;
  progress_percentage: number;
  estimated_remaining_seconds?: number;
  parameters?: {
    exaggeration?: number;
    cfg_weight?: number;
    temperature?: number;
    output_format?: string;
  };
}

export interface LongTextJobList {
  jobs: LongTextJobListItem[];
  total_count: number;
}

export interface LongTextJobCreateResponse {
  job_id: string;
  status: LongTextJobStatus;
  message: string;
  estimated_processing_time_seconds?: number;
}

export interface LongTextRequest {
  text: string;
  voice?: string;
  exaggeration?: number;
  cfg_weight?: number;
  temperature?: number;
  language?: string;
  voice_file?: File;
  output_format?: 'mp3' | 'wav';
  silence_padding_ms?: number;
  chunking_strategy?: 'sentence' | 'paragraph' | 'word' | 'fixed';
  quality_preset?: 'fast' | 'balanced' | 'high';
  chunk_size?: number;
  session_id?: string;
  enable_pauses?: boolean;
  custom_pauses?: Record<string, number>;
}

export interface LongTextSSEEvent {
  event_type: 'progress' | 'chunk_completed' | 'job_completed' | 'job_failed' | 'job_paused' | 'job_resumed' | string;
  job_id: string;
  timestamp: string;
  data: {
    progress?: LongTextProgress;
    chunk_info?: {
      chunk_index: number;
      audio_file_path: string;
      duration_ms: number;
    };
    error?: string;
    message?: string;
  };
}

// History Management Types
export interface LongTextHistoryFilter {
  status?: LongTextJobStatus;
  start_date?: string;
  end_date?: string;
  min_duration?: number;
  max_duration?: number;
  min_text_length?: number;
  max_text_length?: number;
  voice_name?: string;
  search_text?: string;
  is_archived?: boolean;
}

export type LongTextHistorySort =
  | 'created_desc'
  | 'created_asc'
  | 'completed_desc'
  | 'completed_asc'
  | 'name_asc'
  | 'name_desc'
  | 'duration_desc'
  | 'duration_asc'
  | 'size_desc'
  | 'size_asc';

export interface LongTextHistoryStats {
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_audio_duration_seconds: number;
  total_characters_processed: number;
  average_processing_time_seconds: number;
  total_storage_used_bytes: number;
  success_rate_percentage: number;
  most_used_voice?: string;
  date_range: {
    earliest_job: string;
    latest_job: string;
  };
}

export interface LongTextJobDetails {
  job: LongTextJobMetadata;
  progress?: LongTextProgress;
  chunks: LongTextChunk[];
  performance_metrics: {
    total_processing_time_ms: number;
    average_chunk_time_ms: number;
    peak_memory_usage_mb?: number;
    characters_per_second?: number;
  };
  final_audio?: {
    file_path: string;
    file_size_bytes: number;
    duration_seconds: number;
    sample_rate: number;
    channels: number;
  };
}

export interface LongTextJobUpdateRequest {
  display_name?: string;
  tags?: string[];
  is_archived?: boolean;
}

export interface LongTextJobRetryRequest {
  voice?: string;
  exaggeration?: number;
  cfg_weight?: number;
  temperature?: number;
  language?: string;
  output_format?: 'mp3' | 'wav';
  force_restart?: boolean;
}

export type BulkJobAction = 'delete' | 'archive' | 'unarchive' | 'retry' | string;

export interface BulkJobActionRequest {
  action: BulkJobAction;
  job_ids: string[];
  retry_options?: LongTextJobRetryRequest;
  confirm?: boolean;
}

export interface BulkJobActionResponse {
  success_count: number;
  failed_count: number;
  results: Array<{
    job_id: string;
    success: boolean;
    error?: string;
    new_job_id?: string; // For retry action
  }>;
}

export interface LongTextHistoryItem extends LongTextJobListItem {
  display_name?: string;
  tags?: string[];
  completion_timestamp?: string;
  audio_file_size?: number;
  total_duration_seconds?: number;
  retry_count?: number;
  original_job_id?: string;
  is_archived?: boolean;
  last_accessed?: string;
  parameters?: {
    exaggeration?: number;
    cfg_weight?: number;
    temperature?: number;
    output_format?: string;
  };
}

export interface MessageResponse {
  message: string;
}