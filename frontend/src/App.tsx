import React from 'react';
import { Route, Switch } from 'wouter';
import { Github, MessageCircle } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import ThemeToggle from './components/theme-toggle';
import ModelToggle from './components/ModelToggle'; // <-- import
import Navigation from './components/Navigation';
import TTSPage from './pages/TTSPage';
import MemoryPage from './pages/MemoryPage';
import { createTTSService } from './services/tts';
import { useApiEndpoint } from './hooks/useApiEndpoint';
import { getFrontendVersion } from './lib/version';
import { useGlobalErrorHandler } from './hooks/useGlobalErrorHandler';

function App() {
  // Initialize global error handler
  useGlobalErrorHandler();

  const { apiBaseUrl } = useApiEndpoint();
  const ttsService = createTTSService(apiBaseUrl);

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

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header with Navigation and Theme Toggle */}
      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container max-w-6xl mx-auto px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex-1 min-w-fit items-center gap-2">
              <h1 className="text-lg font-semibold min-w-fit">Chatterbox TTS</h1>
            </div>
            <Navigation />
          </div>
           <ModelToggle />   {/* <-- Added button */}
          <ThemeToggle />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl mx-auto w-full">
        <Switch>
          <Route path="/" component={TTSPage} />
          <Route path="/memory-management" component={MemoryPage} />
          <Route>
            {/* 404 Route */}
            <div className="container mx-auto px-4 py-8 text-center">
              <h2 className="text-2xl font-bold mb-2">Page Not Found</h2>
              <p className="text-muted-foreground">The page you're looking for doesn't exist.</p>
            </div>
          </Route>
        </Switch>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card/50">
        <div className="w-full max-w-4xl mx-auto px-4 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-sm text-muted-foreground flex flex-col gap-2">
              <div className="flex flex-row gap-8">
                <span>Frontend v{getFrontendVersion()}</span>
                {apiInfo?.version && <span>API v{apiInfo.version}</span>}
              </div>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/travisvn/chatterbox-tts-api"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors duration-300"
              >
                <Github className="w-4 h-4" />
                <span>GitHub</span>
              </a>
              <a
                href="https://chatterboxtts.com/discord"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors duration-300"
              >
                <MessageCircle className="w-4 h-4" />
                <span>Join Discord</span>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App; 