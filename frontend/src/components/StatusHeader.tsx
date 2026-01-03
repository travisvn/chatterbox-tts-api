import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Activity, Clock, Cpu, Settings, Star, Zap, Globe, Mic } from 'lucide-react';
import type { HealthResponse, TTSProgress, TTSStatistics, VoiceSample } from '../types';

interface StatusHeaderProps {
  health: HealthResponse | undefined;
  progress: TTSProgress | undefined;
  statistics: TTSStatistics | undefined;
  isLoadingHealth: boolean;
  hasErrors: boolean;
  apiVersion?: string;
  progressSettings?: {
    onlyShowMyRequests: boolean;
    onToggleOnlyMyRequests: () => void;
  };
  defaultVoiceSettings?: {
    defaultVoice: string | null;
    voices: VoiceSample[];
    onSetDefaultVoice: (voiceName: string) => Promise<boolean>;
    onClearDefaultVoice: () => Promise<boolean>;
    isLoading: boolean;
  };
}

export default function StatusHeader({
  health,
  progress,
  statistics,
  isLoadingHealth,
  hasErrors,
  apiVersion,
  progressSettings,
  defaultVoiceSettings
}: StatusHeaderProps) {
  const [showSettings, setShowSettings] = useState(false);

  const getStatusInfo = () => {
    if (hasErrors) {
      return {
        color: 'text-destructive',
        icon: <AlertCircle className="w-4 h-4" />,
        status: 'Connection Error',
        subtitle: 'Unable to connect to API'
      };
    }

    if (isLoadingHealth || !health) {
      return {
        color: 'text-muted-foreground',
        icon: <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary" />,
        status: 'Connecting...',
        subtitle: 'Establishing connection'
      };
    }

    // Handle initialization states
    if (health.initialization_state === 'initializing') {
      return {
        color: 'text-blue-600 dark:text-blue-400',
        icon: <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 dark:border-blue-400" />,
        status: 'Initializing Model',
        subtitle: health.initialization_progress || 'Loading TTS model...'
      };
    }

    if (health.initialization_state === 'error') {
      return {
        color: 'text-destructive',
        icon: <AlertCircle className="w-4 h-4" />,
        status: 'Initialization Failed',
        subtitle: health.initialization_error || 'Model failed to load'
      };
    }

    if (health.initialization_state === 'starting' || health.status === 'starting') {
      return {
        color: 'text-yellow-600 dark:text-yellow-400',
        icon: <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 dark:border-yellow-400" />,
        status: 'Starting Up',
        subtitle: 'Server is starting...'
      };
    }

    if (progress?.is_processing) {
      return {
        color: 'text-green-600 dark:text-green-400',
        icon: <Activity className="w-4 h-4 animate-pulse" />,
        status: 'Processing',
        subtitle: progress.current_step || progress.status.replace('_', ' ')
      };
    }

    if (!health.model_loaded) {
      return {
        color: 'text-yellow-600 dark:text-yellow-400',
        icon: <AlertCircle className="w-4 h-4" />,
        status: 'Model Loading',
        subtitle: `${health.device} - initializing`
      };
    }

    return {
      color: 'text-green-600 dark:text-green-400',
      icon: <CheckCircle className="w-4 h-4" />,
      status: 'Ready',
      subtitle: `${health.device} - ${statistics?.total_requests || 0} requests`
    };
  };

  const statusInfo = getStatusInfo();

  const getModelBadge = () => {
    if (!health?.model_loaded) return null;
    
    const modelType = health.model_type || health.model_capabilities?.model_type;
    
    if (!modelType) return null;

    switch (modelType) {
      case 'turbo':
        return {
          label: 'Turbo',
          icon: <Zap className="w-3 h-3" />,
          className: 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800'
        };
      case 'multilingual':
        return {
          label: 'Multilingual',
          icon: <Globe className="w-3 h-3" />,
          className: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800'
        };
      case 'standard':
        return {
          label: 'Standard',
          icon: <Mic className="w-3 h-3" />,
          className: 'bg-gray-100 dark:bg-gray-800/40 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700'
        };
      default:
        return null;
    }
  };

  const modelBadge = getModelBadge();

  const formatUptime = () => {
    if (!statistics) return null;

    if (statistics.total_requests > 0) {
      return (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>{statistics.total_requests} total</span>
          <span>•</span>
          <span>{statistics.success_rate.toFixed(1)}% success</span>
          {statistics.average_duration_seconds > 0 && (
            <>
              <span>•</span>
              <span>{statistics.average_duration_seconds.toFixed(1)}s avg</span>
            </>
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="text-center relative">
      <h1 className="text-4xl font-bold text-foreground mb-2">
        🎤 Chatterbox TTS
      </h1>
      <p className="text-muted-foreground mb-3">
        Convert text to speech with voice cloning
      </p>

      {/* Enhanced Status Display */}
      <div className="inline-flex flex-col items-center gap-1">
        <div className={`inline-flex items-center gap-2 text-sm ${statusInfo.color}`}>
          {statusInfo.icon}
          <span className="font-medium">{statusInfo.status}</span>
          <span>•</span>
          <span>{statusInfo.subtitle}</span>
          {modelBadge && (
            <>
              <span>•</span>
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${modelBadge.className}`}>
                {modelBadge.icon}
                {modelBadge.label}
              </span>
            </>
          )}
          {apiVersion && (
            <>
              <span>•</span>
              <span className="text-muted-foreground">API v{apiVersion}</span>
            </>
          )}
        </div>

        {formatUptime()}

        {/* Current Progress (if processing) */}
        {progress?.is_processing && progress.progress_percentage !== undefined && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
            <Clock className="w-3 h-3" />
            <span>{progress.progress_percentage.toFixed(1)}% complete</span>
            {progress.current_chunk && progress.total_chunks && (
              <>
                <span>•</span>
                <span>{progress.current_chunk}/{progress.total_chunks} chunks</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Settings Button */}
      {(progressSettings || defaultVoiceSettings) && (
        <div className="absolute right-0 sm:right-4 top-12 flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="relative p-2 rounded-lg hover:bg-accent border border-border bg-card duration-300 dark:shadow-sm dark:hover:shadow-md cursor-pointer"
              title="Settings"
            >
              <Settings className="w-5 h-5 text-foreground" />
              <span className="sr-only">Settings</span>
            </button>

            {showSettings && (
              <div className="absolute top-full right-0 mt-2 bg-card border border-border rounded-lg shadow-lg p-4 w-[300px] sm:w-[500px] max-w-[70vw] z-10">
                <h3 className="text-sm font-medium text-foreground mb-3">Settings</h3>

                <div className="space-y-4">
                  {/* Progress Settings */}
                  {progressSettings && (
                    <div>
                      <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide text-left">Progress</h4>
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={progressSettings.onlyShowMyRequests}
                          onChange={progressSettings.onToggleOnlyMyRequests}
                          className="w-4 h-4 text-primary border-border rounded focus:ring-primary focus:ring-2"
                        />
                        <div className="text-left">
                          <div className="text-sm font-medium text-foreground">
                            Only show my requests
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Only display progress for TTS requests made from this browser tab
                          </div>
                        </div>
                      </label>
                    </div>
                  )}

                  {/* Default Voice Settings */}
                  {defaultVoiceSettings && (
                    <div>
                      <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide text-left">Default Voice</h4>
                      {defaultVoiceSettings.isLoading ? (
                        <div className="text-sm text-muted-foreground">Loading voices...</div>
                      ) : defaultVoiceSettings.voices.length === 0 ? (
                        <div className="text-sm text-muted-foreground">No voices available</div>
                      ) : (
                        <div className="space-y-2">
                          <div className="text-xs text-muted-foreground mb-2 text-left">
                            Select which voice to use as default when no voice is specified
                          </div>
                          <div className="max-h-32 overflow-y-auto space-y-1">
                            {/* None/System Default */}
                            <label className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-accent">
                              <input
                                type="radio"
                                name="defaultVoice"
                                checked={!defaultVoiceSettings.defaultVoice}
                                onChange={() => defaultVoiceSettings.onClearDefaultVoice()}
                                className="w-3 h-3 text-primary"
                              />
                              <div className="flex items-center gap-2">
                                <span className="text-sm text-muted-foreground">(System Default)</span>
                              </div>
                            </label>
                            {/* Voice Options */}
                            {defaultVoiceSettings.voices.map((voice) => (
                              <label key={voice.id} className="flex items-center gap-2 cursor-pointer p-2 rounded hover:bg-accent">
                                <input
                                  type="radio"
                                  name="defaultVoice"
                                  checked={defaultVoiceSettings.defaultVoice === voice.name}
                                  onChange={() => defaultVoiceSettings.onSetDefaultVoice(voice.name)}
                                  className="w-3 h-3 text-primary"
                                />
                                <div className="flex items-center gap-2">
                                  <span className="text-sm text-foreground">{voice.name}</span>
                                  {defaultVoiceSettings.defaultVoice === voice.name && (
                                    <Star className="w-3 h-3 text-yellow-500 fill-current" />
                                  )}
                                </div>
                              </label>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-3 border-t border-border">
                  <button
                    onClick={() => setShowSettings(false)}
                    className="text-xs text-primary hover:text-primary/80 transition-colors duration-200 w-full text-center"
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Click outside to close settings */}
      {showSettings && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setShowSettings(false)}
        />
      )}
    </div>
  );
} 