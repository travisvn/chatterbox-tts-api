import React, { useState, useMemo } from 'react';
import { Volume2 } from 'lucide-react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Button } from '../components/ui/button';
import {
  ApiEndpointSelector,
  TextInput,
  AdvancedSettings,
  AudioPlayer,
  LongTextHistory
} from '../components/tts';
import VoiceLibrary from '../components/VoiceLibrary';
import AudioHistory from '../components/AudioHistory';
import StatusHeader from '../components/StatusHeader';
import StatusProgressOverlay from '../components/StatusProgressOverlay';
import StatusStatisticsPanel from '../components/StatusStatisticsPanel';
import StreamingProgressComponent from '../components/tts/StreamingProgress';
import LongTextProgress from '../components/tts/LongTextProgress';
import LongTextJobs from '../components/tts/LongTextJobs';
import { createTTSService } from '../services/tts';
import { createLongTextTTSService } from '../services/longTextTTS';
import { useApiEndpoint } from '../hooks/useApiEndpoint';
import { useVoiceLibrary } from '../hooks/useVoiceLibrary';
import { useAudioHistory } from '../hooks/useAudioHistory';
import { useAdvancedSettings } from '../hooks/useAdvancedSettings';
import { useTextInput } from '../hooks/useTextInput';
import { useStatusMonitoring } from '../hooks/useStatusMonitoring';
import { useProgressSettings } from '../hooks/useProgressSettings';
import { useDefaultVoice } from '../hooks/useDefaultVoice';
import { useStreamingTTS } from '../hooks/useStreamingTTS';
import { useLongTextTTS } from '../hooks/useLongTextTTS';
import { useLongTextHistory } from '../hooks/useLongTextHistory';
import { useHistoryTab } from '../hooks/useHistoryTab';
import type { TTSRequest, LongTextRequest } from '../types';

export default function TTSPage() {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isClickedGenerating, setIsClickedGenerating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showStatistics, setShowStatistics] = useState(false);

  // API endpoint management
  const { apiBaseUrl, updateApiBaseUrl } = useApiEndpoint();

  // Text input management with persistence
  const { text, updateText, clearText, hasText } = useTextInput();

  // Advanced settings management with persistence
  const {
    exaggeration,
    cfgWeight,
    temperature,
    updateExaggeration,
    updateCfgWeight,
    updateTemperature,
    resetToDefaults,
    isDefault
  } = useAdvancedSettings();

  // Progress settings and session tracking
  const {
    settings: progressSettings,
    updateSettings: updateProgressSettings,
    trackRequest,
    shouldShowProgress,
    dismissProgress,
    isLongTextRequest,
    sessionId
  } = useProgressSettings();

  // Streaming TTS management
  const {
    isStreaming,
    progress: streamingProgress,
    audioUrl: streamingAudioUrl,
    error: streamingError,
    audioInfo,
    isStreamingEnabled,
    toggleStreaming,
    streamingFormat,
    setStreamingFormat,
    startStreaming,
    stopStreaming,
    clearAudio: clearStreamingAudio
  } = useStreamingTTS({
    apiBaseUrl,
    sessionId
  });

  // Long text TTS management
  const {
    currentJob,
    progress: longTextProgress,
    isJobActive,
    error: longTextError,
    audioUrl: longTextAudioUrl,
    jobList,
    totalJobCount,
    isLoadingJobs,
    isSubmitting,
    submitJob,
    pauseJob,
    resumeJob,
    cancelJob,
    refetchJobs,
    estimateProcessingTime,
    shouldUseLongText,
    getStatusMessage
  } = useLongTextTTS({
    apiBaseUrl,
    sessionId
  });

  // Voice library management with backend health monitoring
  const {
    voices,
    selectedVoice,
    setSelectedVoice,
    addVoice,
    deleteVoice,
    renameVoice,
    refreshVoices,
    addAlias,
    removeAlias,
    isLoading: voicesLoading,
    isBackendReady: voicesBackendReady,
    error: voicesError
  } = useVoiceLibrary();

  // Audio history management
  const {
    audioHistory,
    addAudioRecord,
    deleteAudioRecord,
    renameAudioRecord,
    clearHistory,
    isLoading: historyLoading
  } = useAudioHistory();

  // Long text history management
  const {
    jobs: longTextJobs,
    totalCount: longTextTotalCount,
    currentPage: longTextCurrentPage,
    totalPages: longTextTotalPages,
    selectedJobs: longTextSelectedJobs,
    isLoadingHistory: isLoadingLongTextHistory,
    isLoadingStats: isLoadingLongTextStats,
    stats: longTextStats,
    settings: longTextHistorySettings,
    updateJob: updateLongTextJob,
    retryJob: retryLongTextJob,
    deleteJob: deleteLongTextJob,
    archiveJob: archiveLongTextJob,
    unarchiveJob: unarchiveLongTextJob,
    getAudioUrl: getLongTextAudioUrl,
    downloadAudio: downloadLongTextAudio,
    bulkDelete: bulkDeleteLongTextJobs,
    bulkArchive: bulkArchiveLongTextJobs,
    bulkUnarchive: bulkUnarchiveLongTextJobs,
    bulkRetry: bulkRetryLongTextJobs,
    toggleJobSelection: toggleLongTextJobSelection,
    selectAllJobs: selectAllLongTextJobs,
    clearSelection: clearLongTextSelection,
    goToPage: goToLongTextPage,
    nextPage: nextLongTextPage,
    prevPage: prevLongTextPage,
    searchJobs: searchLongTextJobs,
    updateSort: updateLongTextSort,
    clearHistory: clearLongTextHistory,
    refetchHistory: refetchLongTextHistory,
    updateSettings: updateLongTextHistorySettings
  } = useLongTextHistory({
    apiBaseUrl,
    sessionId
  });

  // Default voice management with backend health monitoring
  const {
    defaultVoice,
    updateDefaultVoice,
    clearDefaultVoice,
    isLoading: defaultVoiceLoading,
    isBackendReady: defaultVoiceBackendReady,
    healthStatus
  } = useDefaultVoice();

  // History tab selection with persistence
  const { historyTab, updateHistoryTab } = useHistoryTab();

  // Create TTS service with current API base URL and session ID
  const ttsService = useMemo(() => createTTSService(apiBaseUrl, sessionId), [apiBaseUrl, sessionId]);

  // Status monitoring with real-time updates
  const {
    progress,
    statistics,
    isProcessing,
    hasError: statusHasError,
    isLoadingStats
  } = useStatusMonitoring(apiBaseUrl);

  const { data: health, isLoading: isLoadingHealth } = useQuery({
    queryKey: ['health', apiBaseUrl],
    queryFn: ttsService.getHealth,
    refetchInterval: 3000, // More frequent during startup
    retry: true,
    retryDelay: 1000
  });

  // Fetch API info (including version) periodically
  const { data: apiInfo } = useQuery({
    queryKey: ['apiInfo', apiBaseUrl],
    queryFn: async () => {
      const response = await fetch(`${apiBaseUrl}/info`);
      if (!response.ok) throw new Error('Failed to fetch API info');
      return response.json();
    },
    refetchInterval: 60000, // Refresh every minute
    retry: false
  });

  // Standard (non-streaming) generation mutation
  const generateMutation = useMutation({
    mutationFn: ttsService.generateSpeech,
    onMutate: (variables) => {
      // Track this request as originating from this frontend
      if (variables.session_id) {
        trackRequest(variables.session_id);
      }
    },
    onSuccess: async (audioBlob) => {
      // Clean up previous audio URL
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      // Create new audio URL
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

      // Save to audio history
      try {
        await addAudioRecord(
          audioBlob,
          {
            text,
            exaggeration,
            cfgWeight,
            temperature,
            voiceId: selectedVoice?.id,
            voiceName: selectedVoice?.name || defaultVoice || "Default"
          }
        );
      } catch (error) {
        console.error('Failed to save audio record:', error);
      }
    },
    onError: (error) => {
      console.error('TTS generation failed:', error);
      alert('Failed to generate speech. Please try again.');
    }
  });

  const handleGenerate = async () => {
    if (!text.trim()) {
      alert('Please enter some text to convert to speech.');
      return;
    }

    setIsClickedGenerating(true);

    // Check if we should use long text processing
    if (shouldUseLongText(text)) {

      setTimeout(() => {
        setIsClickedGenerating(false);
      }, 8000);

      // Use long text TTS
      const longTextRequest: LongTextRequest = {
        text,
        voice: selectedVoice?.name,
        exaggeration,
        cfg_weight: cfgWeight,
        temperature,
        language: 'en',
        output_format: 'mp3',
        session_id: sessionId
      };

      if (selectedVoice?.file) {
        longTextRequest.voice_file = selectedVoice.file;
      }

      try {
        // Track this request as long-text type
        trackRequest(sessionId, 'long-text');
        submitJob(longTextRequest);
      } catch (error) {
        console.error('Long text TTS failed:', error);
        alert('Failed to start long text processing. Please try again.');
      }
      return;
    } else {
      setTimeout(() => {
        setIsClickedGenerating(false);
      }, 4000);
    }

    // Prepare request data for standard/streaming TTS
    const requestData: TTSRequest = {
      input: text,
      exaggeration,
      cfg_weight: cfgWeight,
      temperature,
      session_id: sessionId
    };

    if (selectedVoice) {
      // Use voice name for backend voice library
      requestData.voice = selectedVoice.name;

      // Also include voice file if it's a client-side voice (for backward compatibility)
      if (selectedVoice.file) {
        requestData.voice_file = selectedVoice.file;
      }
    }

    // Track this request
    trackRequest(sessionId);

    if (isStreamingEnabled) {
      // Use streaming
      try {
        await startStreaming(requestData);

        // If streaming completes successfully and we have a final audio URL, save to history
        if (streamingAudioUrl) {
          try {
            const response = await fetch(streamingAudioUrl);
            const audioBlob = await response.blob();

            await addAudioRecord(
              audioBlob,
              {
                text,
                exaggeration,
                cfgWeight,
                temperature,
                voiceId: selectedVoice?.id,
                voiceName: selectedVoice?.name || defaultVoice || "Default"
              }
            );
          } catch (error) {
            console.error('Failed to save streaming audio to history:', error);
          }
        }
      } catch (error) {
        console.error('Streaming failed:', error);
        alert('Failed to stream speech. Please try again.');
      }
    } else {
      // Use standard generation
      generateMutation.mutate(requestData);
    }
  };

  const isBackendReady = !!apiInfo; // If /info returned successfully, backend is ready
// const isInitializing = !apiInfo; // If /info is not fetched yet, consider initializing

  const isInitializing = healthStatus === 'initializing' || health?.status === 'initializing';

  // Determine if generation is in progress (streaming, standard, or long text)
  const isGenerating = isClickedGenerating || generateMutation.isPending || isStreaming || isSubmitting || isJobActive;

  // Use long text audio URL if available, then streaming, then standard
  const currentAudioUrl = longTextAudioUrl || streamingAudioUrl || audioUrl;

  // Check if current text requires long text processing
  const isLongText = shouldUseLongText(text);
  const estimatedTime = isLongText ? estimateProcessingTime(text.length) : null;

  return (
    <>
      <div className="container mx-auto px-4 py-8 flex flex-col items-center justify-center gap-4">
        {/* Status Header */}
        <div className="flex justify-between items-start w-full max-w-6xl mx-auto relative">
          <div className="flex-1">
            <StatusHeader
              health={health}
              progress={progress}
              statistics={statistics}
              isLoadingHealth={isLoadingHealth}
              hasErrors={statusHasError}
              apiVersion={apiInfo?.version || apiInfo?.api_version}
              progressSettings={{
                onlyShowMyRequests: progressSettings.onlyShowMyRequests,
                onToggleOnlyMyRequests: () => updateProgressSettings({
                  onlyShowMyRequests: !progressSettings.onlyShowMyRequests
                })
              }}
              defaultVoiceSettings={{
                defaultVoice,
                voices,
                onSetDefaultVoice: updateDefaultVoice,
                onClearDefaultVoice: clearDefaultVoice,
                isLoading: voicesLoading || defaultVoiceLoading
              }}
            />
          </div>
        </div>

{/* Backend Loading / Model State */}
{(!isBackendReady || isInitializing) && (
  <div className="w-full max-w-2xl mx-auto">
    <div className="bg-yellow-100 border border-yellow-300 rounded-lg p-4 text-center">
      <div className="flex items-center justify-center gap-2 mb-2">
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600"></div>
        <span className="text-sm font-medium text-yellow-700">
          {isInitializing
            ? 'Backend initializing...'
            : 'Loading voice library...'
          }
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        {isInitializing
          ? 'TTS model is starting up. Voice library will load once ready.'
          : 'Connecting to voice library and loading default settings.'
        }
      </p>
    </div>
  </div>
)}


        <div className="w-full max-w-3xl mx-auto flex flex-col items-center justify-center gap-4">

          <div className="flex flex-col items-center justify-center gap-2 w-full">
            <button
              onClick={() => setShowStatistics(!showStatistics)}
              className="px-3 py-1 text-xs bg-primary/10 hover:bg-primary/20 text-primary rounded-full transition-colors duration-300"
            >
              {showStatistics ? 'Hide Stats' : 'Show Stats'}
            </button>
            {/* Statistics Panel (collapsible) */}
            {showStatistics && (
              <StatusStatisticsPanel
                statistics={statistics}
                isLoading={isLoadingStats}
                hasError={statusHasError}
              />
            )}
          </div>

          <div className="max-w-3xl mx-auto gap-4 flex flex-col w-full">
            {/* API Endpoint Selector */}
            <div className="">
              <ApiEndpointSelector
                apiBaseUrl={apiBaseUrl}
                onUrlChange={updateApiBaseUrl}
              />
            </div>

            {/* Text Input */}
            <TextInput
              value={text}
              onChange={updateText}
              onClear={clearText}
              hasText={hasText}
              isStreamingEnabled={isStreamingEnabled}
              onToggleStreaming={toggleStreaming}
            />

            {/* Long Text Detection */}
            {isLongText && (
              <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center mt-0.5">
                    <span className="text-white text-xs font-bold">!</span>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-1">
                      Long Text Detected ({text.length.toLocaleString()} characters)
                    </h4>
                    <p className="text-xs text-blue-700 dark:text-blue-300 mb-2">
                      This text will be processed using long text TTS mode. Your text will be intelligently split into chunks and processed in the background.
                    </p>
                    {estimatedTime && (
                      <p className="text-xs text-blue-600 dark:text-blue-400">
                        Estimated processing time: ~{Math.ceil(estimatedTime / 60)} minutes
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Streaming Progress */}
            {(isStreaming || streamingProgress || streamingAudioUrl || streamingError) && (
              <StreamingProgressComponent
                isStreaming={isStreaming}
                progress={streamingProgress}
                audioUrl={streamingAudioUrl}
                error={streamingError}
                audioInfo={audioInfo}
                onStop={stopStreaming}
                onClear={clearStreamingAudio}
              />
            )}

            {/* Long Text Progress */}
            {(isJobActive || currentJob || longTextError) && (
              <LongTextProgress
                job={currentJob}
                progress={longTextProgress}
                isJobActive={isJobActive}
                audioUrl={longTextAudioUrl}
                error={longTextError}
                onPause={pauseJob}
                onResume={resumeJob}
                onCancel={cancelJob}
                onDownload={(url) => {
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = `chatterbox-long-text-${Date.now()}.mp3`;
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                }}
                isPausing={false}
                isResuming={false}
                isCancelling={false}
              />
            )}

            {/* Voice Library */}
            <VoiceLibrary
              voices={voices}
              selectedVoice={selectedVoice}
              onVoiceSelect={setSelectedVoice}
              onAddVoice={addVoice}
              onDeleteVoice={deleteVoice}
              onRenameVoice={renameVoice}
              onRefresh={refreshVoices}
              isLoading={voicesLoading}
              defaultVoice={defaultVoice}
              onSetDefaultVoice={updateDefaultVoice}
              onClearDefaultVoice={clearDefaultVoice}
              onAddAlias={addAlias}
              onRemoveAlias={removeAlias}
            />

            {/* Voice Library Error Display */}
            {voicesError && !voicesLoading && (
              <div className="text-center py-4 text-muted-foreground">
                <p className="text-sm text-destructive mb-2">
                  Failed to load voice library: {voicesError.message || 'Unknown error'}
                </p>
                <button
                  onClick={refreshVoices}
                  className="text-xs text-primary hover:text-primary/80 underline"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Advanced Settings */}
            <AdvancedSettings
              showAdvanced={showAdvanced}
              onToggle={() => setShowAdvanced(!showAdvanced)}
              exaggeration={exaggeration}
              onExaggerationChange={updateExaggeration}
              cfgWeight={cfgWeight}
              onCfgWeightChange={updateCfgWeight}
              temperature={temperature}
              onTemperatureChange={updateTemperature}
              onResetToDefaults={resetToDefaults}
              isDefault={isDefault}
            />

            {/* Current Voice Indicator */}
            {selectedVoice && (
              <div className="text-center text-sm text-muted-foreground">
                Using voice: <span className="font-medium text-foreground">{selectedVoice.name}</span>
                {defaultVoice === selectedVoice.name && (
                  <span className="ml-2 text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded">
                    Default
                  </span>
                )}
              </div>
            )}

            {/* Generate Button */}
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || !hasText}
              className="w-full py-6 px-6 text-xl [&_svg]:size-6 [&_svg:not([class*='size-'])]:size-6 flex gap-4"
            >
              <Volume2 className="w-5 h-5 mr-2" />
              {isGenerating ? (isStreaming ? 'Streaming...' : 'Generating...') : 'Generate Speech'}
            </Button>

            {/* Audio Player - Only show for non-streaming audio or completed streaming */}
            {currentAudioUrl && !isStreaming && (
              <AudioPlayer audioUrl={currentAudioUrl} />
            )}
          </div>
        </div>

        {/* Active Jobs Monitor - Only show when there are jobs currently processing */}
        {jobList.length > 0 && (
          <div className="w-full max-w-3xl mx-auto">
            <LongTextJobs
              jobs={jobList}
              totalCount={totalJobCount}
              isLoading={isLoadingJobs}
              onRefresh={refetchJobs}
              onDownload={async (jobId) => {
                try {
                  const service = createLongTextTTSService(apiBaseUrl, sessionId);
                  const audioBlob = await service.downloadJobAudio(jobId);
                  const audioUrl = URL.createObjectURL(audioBlob);

                  const link = document.createElement('a');
                  link.href = audioUrl;
                  link.download = `chatterbox-long-text-${jobId.slice(-8)}.mp3`;
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);

                  URL.revokeObjectURL(audioUrl);
                } catch (error) {
                  console.error('Failed to download job:', error);
                  alert('Failed to download audio file');
                }
              }}
              onResume={resumeJob}
              onPause={pauseJob}
              onCancel={cancelJob}
            />
          </div>
        )}

        {/* History Section with Tabs */}
        <div className="w-full max-w-3xl mx-auto">
          {/* History Tabs */}
          <div className="flex border-b border-border mb-4">
            <button
              onClick={() => updateHistoryTab('regular')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${historyTab === 'regular'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
            >
              Regular TTS ({audioHistory.length})
            </button>
            <button
              onClick={() => updateHistoryTab('longtext')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${historyTab === 'longtext'
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
            >
              Long Text ({longTextTotalCount})
            </button>
          </div>

          {/* History Content */}
          {historyTab === 'regular' ? (
            <AudioHistory
              audioHistory={audioHistory}
              onDeleteAudioRecord={deleteAudioRecord}
              onRenameAudioRecord={renameAudioRecord}
              onClearHistory={clearHistory}
              onRestoreSettings={(settings) => {
                updateExaggeration(settings.exaggeration);
                updateCfgWeight(settings.cfgWeight);
                updateTemperature(settings.temperature);
              }}
              onRestoreText={updateText}
              isLoading={historyLoading}
            />
          ) : (
            <LongTextHistory
              jobs={longTextJobs}
              totalCount={longTextTotalCount}
              currentPage={longTextCurrentPage}
              totalPages={longTextTotalPages}
              selectedJobs={longTextSelectedJobs}
              isLoading={isLoadingLongTextHistory}
              isLoadingStats={isLoadingLongTextStats}
              stats={longTextStats}
              onUpdateJob={updateLongTextJob}
              onRetryJob={retryLongTextJob}
              onDeleteJob={deleteLongTextJob}
              onArchiveJob={archiveLongTextJob}
              onUnarchiveJob={unarchiveLongTextJob}
              onDownloadAudio={downloadLongTextAudio}
              onGetAudioUrl={getLongTextAudioUrl}
              onBulkDelete={bulkDeleteLongTextJobs}
              onBulkArchive={bulkArchiveLongTextJobs}
              onBulkUnarchive={bulkUnarchiveLongTextJobs}
              onBulkRetry={bulkRetryLongTextJobs}
              onToggleJobSelection={toggleLongTextJobSelection}
              onSelectAllJobs={selectAllLongTextJobs}
              onClearSelection={clearLongTextSelection}
              onGoToPage={goToLongTextPage}
              onNextPage={nextLongTextPage}
              onPrevPage={prevLongTextPage}
              onSearch={searchLongTextJobs}
              onUpdateSort={updateLongTextSort}
              onClearHistory={clearLongTextHistory}
              showArchived={longTextHistorySettings.showArchived}
              onToggleArchived={() => {
                updateLongTextHistorySettings({ showArchived: !longTextHistorySettings.showArchived });
              }}
              currentSort={longTextHistorySettings.sort}
              onRestoreSettings={(settings) => {
                updateExaggeration(settings.exaggeration);
                updateCfgWeight(settings.cfgWeight);
                updateTemperature(settings.temperature);
              }}
              onRestoreText={updateText}
            />
          )}
        </div>
      </div>

      {/* Progress Overlay */}
      {shouldShowProgress(progress?.request_id) && progress && (
        <StatusProgressOverlay
          progress={progress}
          isVisible={shouldShowProgress(progress?.request_id)}
          onDismiss={dismissProgress}
          isLongText={isLongTextRequest(progress?.request_id)}
        />
      )}
    </>
  );
}