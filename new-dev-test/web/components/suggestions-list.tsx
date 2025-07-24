"use client";

import React, { useEffect, useRef } from "react";
import { Card } from "./ui/card";

interface SuggestionsListProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
  onClose: () => void;
  className?: string;
}

export function SuggestionsList({
  suggestions,
  onSelect,
  onClose,
  className = "",
}: SuggestionsListProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [selectedIndex, setSelectedIndex] = React.useState(0);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : prev));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev));
          break;
        case "Enter":
          e.preventDefault();
          if (suggestions[selectedIndex]) {
            onSelect(suggestions[selectedIndex]);
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [suggestions, selectedIndex, onSelect, onClose]);

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (listRef.current && !listRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  // Scroll selected item into view
  useEffect(() => {
    const selectedElement = listRef.current?.children[selectedIndex] as HTMLElement;
    if (selectedElement && selectedElement.scrollIntoView) {
      selectedElement.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedIndex]);

  if (suggestions.length === 0) return null;

  return (
    <Card
      ref={listRef}
      className={`absolute z-10 w-full mt-1 p-2 max-h-60 overflow-y-auto shadow-lg ${className}`}
      role="listbox"
      aria-label="Suggestions"
    >
      {suggestions.map((suggestion, index) => (
        <div
          key={index}
          role="option"
          aria-selected={index === selectedIndex}
          className={`
            px-3 py-2 cursor-pointer rounded transition-colors
            ${index === selectedIndex ? "bg-blue-50 text-blue-900" : "hover:bg-gray-50"}
          `}
          onClick={() => onSelect(suggestion)}
          onMouseEnter={() => setSelectedIndex(index)}
        >
          {suggestion}
        </div>
      ))}
    </Card>
  );
}
