import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function formatTimestamp(timestamp: string | number | Date): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function extractTagsFromMarkdown(content: string): string[] {
  const tagRegex = /#([a-zA-Z]\w*)/g;
  const matches = content.match(tagRegex);
  return matches ? matches.map((tag) => tag.slice(1)) : [];
}
