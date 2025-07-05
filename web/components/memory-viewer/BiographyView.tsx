import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Edit3, Save, X } from "lucide-react";
import type { Agent } from "@/lib/types";

interface BiographyViewProps {
  selectedAgent: Agent | null;
  biography: string;
  editingBiography: boolean;
  onBiographyChange: (value: string) => void;
  onSaveBiography: () => void;
  onEditBiography: (editing: boolean) => void;
}

export function BiographyView({
  selectedAgent,
  biography,
  editingBiography,
  onBiographyChange,
  onSaveBiography,
  onEditBiography,
}: BiographyViewProps) {
  if (!selectedAgent) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-center text-muted-foreground">
            Select an agent to view their biography
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardContent className="p-6 h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Agent Biography</h3>
          {!editingBiography ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onEditBiography(true)}
            >
              <Edit3 className="h-4 w-4 mr-2" />
              Edit
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  onEditBiography(false);
                  onBiographyChange(selectedAgent.biography || "");
                }}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button size="sm" onClick={onSaveBiography}>
                <Save className="h-4 w-4 mr-2" />
                Save Biography
              </Button>
            </div>
          )}
        </div>

        {editingBiography ? (
          <Textarea
            value={biography}
            onChange={(e) => onBiographyChange(e.target.value)}
            placeholder="Enter agent biography..."
            className="flex-1 min-h-[300px] resize-none"
          />
        ) : (
          <div className="flex-1 overflow-y-auto">
            <p className="text-sm whitespace-pre-wrap">
              {biography || "No biography available for this agent."}
            </p>
          </div>
        )}

        {/* Additional agent info */}
        <div className="mt-4 pt-4 border-t">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Agent ID:</span>
              <p className="font-medium">{selectedAgent.id}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Class:</span>
              <p className="font-medium">{selectedAgent.type || "Unknown"}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Status:</span>
              <p className="font-medium">{selectedAgent.status || "Unknown"}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Autonomy:</span>
              <p className="font-medium">Unknown</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
