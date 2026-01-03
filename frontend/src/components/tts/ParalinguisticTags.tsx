import React from 'react';
import { Smile } from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import type { ParalinguisticTag } from '../../types';

interface ParalinguisticTagsProps {
  tags: ParalinguisticTag[];
  onInsertTag: (tag: string) => void;
  disabled?: boolean;
}

const TAG_ICONS: Record<string, string> = {
  '[laugh]': '😂',
  '[chuckle]': '😏',
  '[cough]': '😷',
  '[sigh]': '😮‍💨',
  '[gasp]': '😲',
  '[clear throat]': '🗣️',
};

export default function ParalinguisticTags({
  tags,
  onInsertTag,
  disabled = false
}: ParalinguisticTagsProps) {
  if (!tags || tags.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-2 mb-3">
          <Smile className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Paralinguistic Tags</span>
          <span className="text-xs text-muted-foreground">(Turbo model only)</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {tags.map((tagInfo) => (
            <button
              key={tagInfo.tag}
              onClick={() => onInsertTag(tagInfo.tag)}
              disabled={disabled}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm bg-primary/10 hover:bg-primary/20 text-primary rounded-full transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              title={tagInfo.description}
            >
              <span>{TAG_ICONS[tagInfo.tag] || '🔊'}</span>
              <span>{tagInfo.tag}</span>
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Click to insert tags into your text for expressive speech
        </p>
      </CardContent>
    </Card>
  );
}
