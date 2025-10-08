import React, { useState } from 'react';
import { Settings, Check, X, Edit } from 'lucide-react';
import { Input } from '../ui/input';
import { Card, CardContent } from '../ui/card';

interface ApiEndpointSelectorProps {
  apiBaseUrl: string;
  onUrlChange: (url: string) => void;
}

export default function ApiEndpointSelector({ apiBaseUrl, onUrlChange }: ApiEndpointSelectorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempUrl, setTempUrl] = useState(apiBaseUrl);

  // Detect environment for user info
const isDockerMode = typeof window !== 'undefined' &&
  window.location.hostname !== 'localhost' &&
  window.location.hostname !== '127.0.0.1';

  const handleSave = () => {
    onUrlChange(tempUrl);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setTempUrl(apiBaseUrl);
    setIsEditing(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  return (
    <Card className="p-4">
      <CardContent className="p-0">
        <div className="flex items-center justify-between flex-col gap-2 sm:flex-row">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">API Endpoint:</span>
            <span className={`text-xs px-2 py-1 rounded ${isDockerMode ? 'bg-primary/10 text-primary' : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'}`}>
              {isDockerMode ? 'Docker Mode' : 'Local Mode'}
            </span>
          </div>

          {!isEditing ? (
            <div className="flex items-center gap-2">
              <code className="text-xs bg-muted text-foreground px-2 py-1 rounded">
                {apiBaseUrl}
              </code>
              <button
                onClick={() => setIsEditing(true)}
                className="p-1 hover:bg-accent rounded transition-colors duration-300"
                title="Edit endpoint URL"
              >
                <Edit className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Input
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                onKeyDown={handleKeyPress}
                className="text-xs"
                placeholder="http://localhost:4123/v1"
                autoFocus
              />
              <button
                onClick={handleSave}
                className="p-1 hover:bg-green-100 dark:hover:bg-green-900/30 text-green-600 dark:text-green-400 rounded transition-colors duration-300"
                title="Save"
              >
                <Check className="w-4 h-4" />
              </button>
              <button
                onClick={handleCancel}
                className="p-1 hover:bg-destructive/10 text-destructive rounded transition-colors duration-300"
                title="Cancel"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 