"use client";

import { Info } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AboutButtonProps {
  onClick: () => void;
}

export default function AboutButton({ onClick }: AboutButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
      title="About"
    >
      <Info size={12} />
    </Button>
  );
}
