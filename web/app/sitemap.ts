import { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://freeagentics.com";

  // Static pages
  const staticPages = [
    "",
    "/dashboard",
    "/agents",
    "/about",
    "/features",
    "/docs",
    "/support",
    "/privacy",
    "/terms",
  ].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority: route === "" ? 1 : 0.8,
  }));

  // Dynamic pages (agents, docs, etc.)
  // In production, these would be fetched from the database
  const dynamicPages: MetadataRoute.Sitemap = [];

  return [...staticPages, ...dynamicPages];
}
