import { ConversationPreset } from '../../lib/types';

interface PresetSelectorProps {
  currentPreset: ConversationPreset | null;
  onPresetChange: (preset: ConversationPreset) => void;
  onSavePreset: (name: string) => void;
  onLoadPreset: (preset: ConversationPreset) => void;
}

export default function PresetSelector({ 
  currentPreset, 
  onPresetChange, 
  onSavePreset, 
  onLoadPreset 
}: PresetSelectorProps) {
  return (
    <div className="preset-selector">
      <p>Preset Selector Component</p>
      {currentPreset && (
        <p>Current: {currentPreset.name}</p>
      )}
    </div>
  );
}