import React from 'react';
import { Settings, RotateCcw, AlertCircle } from 'lucide-react';
import { Slider } from '../ui/slider';
import { Card, CardContent } from '../ui/card';
import type { ModelCapabilities } from '../../types';

interface AdvancedSettingsProps {
  showAdvanced: boolean;
  onToggle: () => void;
  exaggeration: number;
  onExaggerationChange: (value: number) => void;
  cfgWeight: number;
  onCfgWeightChange: (value: number) => void;
  temperature: number;
  onTemperatureChange: (value: number) => void;
  onResetToDefaults: () => void;
  isDefault: boolean;
  capabilities?: ModelCapabilities | null;
}

export default function AdvancedSettings({
  showAdvanced,
  onToggle,
  exaggeration,
  onExaggerationChange,
  cfgWeight,
  onCfgWeightChange,
  temperature,
  onTemperatureChange,
  onResetToDefaults,
  isDefault,
  capabilities
}: AdvancedSettingsProps) {
  const supportsExaggeration = capabilities?.supports_exaggeration ?? true;
  const supportsCfgWeight = capabilities?.supports_cfg_weight ?? true;
  const supportsTemperature = capabilities?.supports_temperature ?? true;
  const isTurboModel = capabilities?.model_type === 'turbo';

  const hasAnySliders = supportsExaggeration || supportsCfgWeight || supportsTemperature;

  return (
    <Card className="">
      <CardContent>
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={onToggle}
            className="flex items-center gap-2 text-sm font-medium text-foreground hover:text-foreground/80 transition-colors duration-300"
          >
            <Settings className="w-4 h-4" />
            Advanced Settings
            <span className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded">
              {showAdvanced ? 'Hide' : 'Show'}
            </span>
          </button>

          {showAdvanced && !isDefault && hasAnySliders && (
            <button
              onClick={onResetToDefaults}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors duration-300"
            >
              <RotateCcw className="w-3 h-3" />
              Reset to Defaults
            </button>
          )}
        </div>

        {showAdvanced && (
          <div className="space-y-4">
            {isTurboModel && !hasAnySliders && (
              <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg">
                <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
                <p className="text-sm text-amber-700 dark:text-amber-300">
                  Turbo model uses paralinguistic tags instead of these parameters. Use tags like [laugh] or [sigh] in your text.
                </p>
              </div>
            )}

            {supportsExaggeration && (
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Exaggeration: {exaggeration}
                </label>
                <Slider
                  min={0.25}
                  max={2.0}
                  step={0.05}
                  value={[exaggeration]}
                  onValueChange={(values) => onExaggerationChange(values[0])}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">Controls emotion intensity</p>
              </div>
            )}

            {supportsCfgWeight && (
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Pace (CFG Weight): {cfgWeight}
                </label>
                <Slider
                  min={0.0}
                  max={1.0}
                  step={0.05}
                  value={[cfgWeight]}
                  onValueChange={(values) => onCfgWeightChange(values[0])}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">Controls speech pace</p>
              </div>
            )}

            {supportsTemperature && (
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Temperature: {temperature}
                </label>
                <Slider
                  min={0.05}
                  max={2.0}
                  step={0.05}
                  value={[temperature]}
                  onValueChange={(values) => onTemperatureChange(values[0])}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">Controls randomness/creativity</p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}