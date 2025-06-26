"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Pause, RotateCcw, FastForward } from "lucide-react";

export function SimulationControls() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Simulation Controls</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Button size="sm" variant="outline">
            <Play className="h-4 w-4 mr-1" />
            Play
          </Button>
          <Button size="sm" variant="outline">
            <Pause className="h-4 w-4 mr-1" />
            Pause
          </Button>
          <Button size="sm" variant="outline">
            <RotateCcw className="h-4 w-4 mr-1" />
            Reset
          </Button>
          <Button size="sm" variant="outline">
            <FastForward className="h-4 w-4 mr-1" />
            Speed
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
