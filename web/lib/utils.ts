import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Update the extractTagsFromMarkdown function to also find wiki-style links
export function extractTagsFromMarkdown(markdown: string): string[] {
  // Match both [[tag]] syntax and #tag syntax (including #tag-with-hyphens)
  const tagRegex = /\[\[(.*?)\]\]|#([\w-]+)/g;
  const matches = Array.from(markdown.matchAll(tagRegex));

  if (!matches.length) return [];

  return matches
    .map((match) => (match[1] || match[2]).trim()) // Get the tag from either capture group
    .filter((tag, index, self) => self.indexOf(tag) === index); // Remove duplicates
}

export function formatTimestamp(date: Date | string | number): string {
  try {
    // Handle null and undefined explicitly
    if (date === null || date === undefined) {
      return "Invalid date";
    }

    // Parse the date to ensure consistency
    let validDate: Date;

    if (date instanceof Date) {
      validDate = date;
    } else if (typeof date === "string") {
      // Handle ISO strings with time zones directly
      if (date.includes("T")) {
        validDate = new Date(date);
      } else {
        // Normalize string date formats to avoid timezone issues
        let normalizedDate = date;

        // Convert 2024/01/15 to 2024-01-15
        if (normalizedDate.includes("/")) {
          normalizedDate = normalizedDate.replace(/\//g, "-");
        }

        // Convert "Jan 15, 2024" to "2024-01-15"
        const monthNameMatch = normalizedDate.match(
          /(\w{3})\s+(\d{1,2}),?\s+(\d{4})/,
        );
        if (monthNameMatch) {
          const [, monthName, day, year] = monthNameMatch;
          const monthMap: Record<string, string> = {
            Jan: "01",
            Feb: "02",
            Mar: "03",
            Apr: "04",
            May: "05",
            Jun: "06",
            Jul: "07",
            Aug: "08",
            Sep: "09",
            Oct: "10",
            Nov: "11",
            Dec: "12",
          };
          const month = monthMap[monthName];
          normalizedDate = `${year}-${month}-${day.padStart(2, "0")}`;
        }

        // Convert "15 Jan 2024" to "2024-01-15"
        const dayFirstMatch = normalizedDate.match(
          /(\d{1,2})\s+(\w{3})\s+(\d{4})/,
        );
        if (dayFirstMatch) {
          const [, day, monthName, year] = dayFirstMatch;
          const monthMap: Record<string, string> = {
            Jan: "01",
            Feb: "02",
            Mar: "03",
            Apr: "04",
            May: "05",
            Jun: "06",
            Jul: "07",
            Aug: "08",
            Sep: "09",
            Oct: "10",
            Nov: "11",
            Dec: "12",
          };
          const month = monthMap[monthName];
          normalizedDate = `${year}-${month}-${day.padStart(2, "0")}`;
        }

        validDate = new Date(normalizedDate + "T00:00:00.000Z");
      }
    } else {
      validDate = new Date(date);
    }

    // Check if the date is valid
    if (isNaN(validDate.getTime())) {
      console.warn("Invalid date value:", date);
      return "Invalid date";
    }

    // Use UTC to avoid timezone issues in tests
    return validDate.toISOString().split("T")[0];
  } catch (error) {
    console.error("Error formatting timestamp:", error, date);
    return "Invalid date";
  }
}
