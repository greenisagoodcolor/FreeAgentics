import JSZip from "jszip"
import type { Agent, Conversation, KnowledgeEntry } from "./types"
import type { LLMSettings } from "./llm-settings"
import { createLogger } from "./debug-logger"
import { extractTagsFromMarkdown } from "./utils" // Import the tag extraction utility

// Create a module-specific logger
const logger = createLogger("knowledge-import")

interface ImportOptions {
  mode: "replace" | "new" | "merge" | "settings-only"
  importSettings: boolean
  importApiKeys: boolean
  importConversations: boolean
}

interface ImportResult {
  agents?: Agent[]
  settings?: LLMSettings
  conversations?: Conversation[]
}

/**
 * Parse markdown content into a knowledge entry
 * @param content Markdown content
 * @param fileName File name for metadata extraction
 * @returns KnowledgeEntry object
 */
export function parseMarkdownToKnowledge(content: string, fileName: string): KnowledgeEntry {
  // Extract metadata from the markdown content
  const metadataRegex = /^---\s*\n([\s\S]*?)\n---\s*\n/
  const metadataMatch = content.match(metadataRegex)

  let id = `knowledge-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`
  let title = fileName.replace(/\.md$/, "").replace(/-/g, " ")
  let tags: string[] = []
  let timestamp = new Date()

  // Extract the content without the metadata section
  let knowledgeContent = content

  if (metadataMatch) {
    const metadataStr = metadataMatch[1]
    const metadataLines = metadataStr.split("\n")

    // Parse metadata
    metadataLines.forEach((line) => {
      const [key, value] = line.split(":").map((part) => part.trim())
      if (!key || !value) return

      switch (key.toLowerCase()) {
        case "id":
          id = value
          break
        case "title":
          title = value
          break
        case "tags":
          tags = value.split(",").map((tag) => tag.trim())
          break
        case "timestamp":
        case "created_at":
          try {
            // Ensure we create a valid date
            const parsedDate = new Date(value)
            if (!isNaN(parsedDate.getTime())) {
              timestamp = parsedDate
            } else {
              logger.warn(`Invalid date format in metadata: ${value}, using current date instead`)
            }
          } catch (e) {
            logger.warn(`Error parsing date in metadata: ${value}, using current date instead`, e)
          }
          break
      }
    })

    // Remove metadata section from content
    knowledgeContent = content.replace(metadataRegex, "").trim()
  }

  // Check for a title in the first heading of the markdown content
  const titleMatch = knowledgeContent.match(/^# (.+)$/m)
  if (titleMatch) {
    // Use the heading as the title
    title = titleMatch[1].trim()

    // Remove the heading from the content to avoid duplication
    knowledgeContent = knowledgeContent.replace(/^# .+$/m, "").trim()
  }

  // Extract tags from the content using the [[tag]] syntax
  const extractedTags = extractTagsFromMarkdown(knowledgeContent)
  if (extractedTags.length > 0) {
    // Merge with any tags from metadata, removing duplicates
    const allTags = [...new Set([...tags, ...extractedTags])]
    tags = allTags
  }

  logger.debug(`Parsed knowledge entry: ${title}`, {
    id,
    tagsCount: tags.length,
    tags: tags.join(", "),
    timestamp: timestamp.toISOString(),
  })

  return {
    id,
    title,
    content: knowledgeContent,
    timestamp,
    tags,
  }
}

/**
 * Import agents, settings, and conversations from a ZIP file
 * @param file ZIP file to import
 * @param options Import options
 * @returns Imported agents, settings, and conversations
 */
export async function importAgentsAndSettingsFromZip(file: File, options: ImportOptions): Promise<ImportResult> {
  try {
    logger.info("Starting import from ZIP file", { fileName: file.name, options })

    // Load the ZIP file
    const zip = new JSZip()
    const zipContent = await zip.loadAsync(file)

    // Initialize the result
    const result: ImportResult = {}

    // Import settings if requested
    if (options.importSettings) {
      const settingsFile = zipContent.files["settings.json"]
      if (settingsFile) {
        const settingsJson = await settingsFile.async("string")
        const settings = JSON.parse(settingsJson) as LLMSettings

        // Handle API key if present and requested
        if (options.importApiKeys && settings.apiKey) {
          try {
            logger.info("Storing imported API key securely", { provider: settings.provider })

            // Use the API endpoint instead of calling storeApiKey directly
            const response = await fetch("/api/api-key/store", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                provider: settings.provider,
                apiKey: settings.apiKey,
              }),
            })

            if (!response.ok) {
              logger.error(`Error storing imported API key: HTTP ${response.status}`)
              throw new Error(`Failed to store API key: HTTP ${response.status}`)
            }

            const data = await response.json()

            if (!data.success) {
              logger.error("Failed to store imported API key:", data.message)
              throw new Error(`Failed to store API key: ${data.message}`)
            }

            // Get the session ID from the response
            const sessionId = data.sessionId

            if (sessionId) {
              // Store the session ID in localStorage (just like manual process)
              localStorage.setItem(`api_session_${settings.provider}`, sessionId)
              logger.info(`Stored session ID in localStorage with key: api_session_${settings.provider}`)

              // Update the settings with the session ID
              settings.apiKeySessionId = sessionId

              // Remove the raw API key from the settings
              delete settings.apiKey

              logger.info("Successfully stored imported API key", {
                provider: settings.provider,
                sessionId,
              })
            } else {
              logger.error("Failed to store imported API key - no session ID returned")
            }
          } catch (error) {
            logger.error("Error storing imported API key", error)
            // Continue with import even if API key storage fails
          }
        } else if (settings.apiKey) {
          // If not importing API keys, remove them from the settings
          logger.info("Removing API key from imported settings (not requested)")
          delete settings.apiKey
        }

        result.settings = settings
        logger.info("Imported settings", { provider: settings.provider, model: settings.model })
      } else {
        logger.warn("No settings.json found in the ZIP file")
      }
    }

    // Skip agent import if in settings-only mode
    if (options.mode === "settings-only") {
      logger.info("Settings-only mode, skipping agent import")
      return result
    }

    // Import agents
    const agentFiles = Object.keys(zipContent.files).filter(
      (path) => path.startsWith("agents/") && path !== "agents/" && path.endsWith(".json"),
    )

    if (agentFiles.length === 0) {
      logger.warn("No agent files found in the ZIP file")
      return result
    }

    // Process each agent file
    const agents: Agent[] = []
    for (const agentFile of agentFiles) {
      const agentJson = await zipContent.files[agentFile].async("string")
      const agent = JSON.parse(agentJson) as Agent

      // Look for knowledge entries for this agent
      const knowledgeFolder = `knowledge/${agent.id}/`
      const knowledgeFiles = Object.keys(zipContent.files).filter(
        (path) => path.startsWith(knowledgeFolder) && path !== knowledgeFolder && path.endsWith(".md"),
      )

      // Process each knowledge file
      const knowledge: KnowledgeEntry[] = []
      for (const knowledgeFile of knowledgeFiles) {
        try {
          const markdownContent = await zipContent.files[knowledgeFile].async("string")
          const fileName = knowledgeFile.split("/").pop() || ""
          const fileId = fileName.replace(".md", "")

          // Use the parseMarkdownToKnowledge function which now extracts titles from headings
          const parsedEntry = parseMarkdownToKnowledge(markdownContent, fileName)

          // Preserve the file ID if no ID was found in the metadata or use the parsed ID
          const entryId = parsedEntry.id.startsWith("knowledge-") ? fileId : parsedEntry.id

          // Create the knowledge entry with all metadata including tags
          knowledge.push({
            ...parsedEntry,
            id: entryId,
          })

          logger.debug(`Imported knowledge entry: ${parsedEntry.title}`, {
            id: entryId,
            tagsCount: parsedEntry.tags.length,
            tags: parsedEntry.tags.join(", "),
          })
        } catch (error) {
          logger.error(`Error parsing knowledge file ${knowledgeFile}:`, error)
          // Continue with other knowledge files even if one fails
        }
      }

      // Add the knowledge entries to the agent
      agent.knowledge = knowledge
      agents.push(agent)
    }

    result.agents = agents
    logger.info(`Imported ${agents.length} agents`)

    // Import conversations if requested
    if (options.importConversations) {
      const conversationFiles = Object.keys(zipContent.files).filter(
        (path) => path.startsWith("conversations/") && path !== "conversations/" && path.endsWith(".json"),
      )

      if (conversationFiles.length > 0) {
        const conversations: Conversation[] = []

        for (const conversationFile of conversationFiles) {
          try {
            const conversationJson = await zipContent.files[conversationFile].async("string")
            const conversation = JSON.parse(conversationJson) as Conversation

            // Ensure dates are properly parsed
            conversation.startTime = new Date(conversation.startTime)
            if (conversation.endTime) {
              conversation.endTime = new Date(conversation.endTime)
            }

            // Ensure message timestamps are properly parsed
            conversation.messages = conversation.messages.map((message) => ({
              ...message,
              timestamp: new Date(message.timestamp),
            }))

            conversations.push(conversation)
          } catch (error) {
            logger.error(`Error parsing conversation file ${conversationFile}:`, error)
            // Continue with other conversations even if one fails
          }
        }

        result.conversations = conversations
        logger.info(`Imported ${conversations.length} conversations`)
      } else {
        logger.warn("No conversation files found in the ZIP file")
      }
    }

    return result
  } catch (error) {
    logger.error("Error importing from ZIP:", error)
    throw new Error(`Failed to import from ZIP: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/**
 * Merge imported agents with existing agents based on the selected mode
 * @param importedAgents Imported agents
 * @param existingAgents Existing agents
 * @param options Import options
 * @returns Merged agents
 */
export function mergeImportedAgents(
  importedAgents: Agent[],
  existingAgents: Agent[],
  options: { mode: "replace" | "new" | "merge" },
): Agent[] {
  // Create a map of existing agents by ID for quick lookup
  const existingAgentsMap = new Map<string, Agent>()
  existingAgents.forEach((agent) => existingAgentsMap.set(agent.id, agent))

  // Process imported agents based on the selected mode
  switch (options.mode) {
    case "replace":
      // Replace existing agents with the same ID, keep others
      return existingAgents
        .map((agent) => {
          const importedAgent = importedAgents.find((a) => a.id === agent.id)
          return importedAgent || agent
        })
        .concat(
          // Add imported agents that don't exist yet
          importedAgents.filter((agent) => !existingAgentsMap.has(agent.id)),
        )

    case "new":
      // Add imported agents with new IDs to avoid conflicts
      const maxId = Math.max(...existingAgents.map((a) => Number.parseInt(a.id) || 0), 0)
      return [
        ...existingAgents,
        ...importedAgents.map((agent, index) => ({
          ...agent,
          id: `${maxId + index + 1}`,
          inConversation: false,
        })),
      ]

    case "merge":
      // Merge knowledge from imported agents into existing agents with the same ID
      return existingAgents
        .map((agent) => {
          const importedAgent = importedAgents.find((a) => a.id === agent.id)
          if (!importedAgent) return agent

          // Create a set of existing knowledge entry IDs for quick lookup
          const existingKnowledgeIds = new Set(agent.knowledge.map((k) => k.id))

          // Merge knowledge entries, avoiding duplicates
          const mergedKnowledge = [
            ...agent.knowledge,
            ...importedAgent.knowledge.filter((k) => !existingKnowledgeIds.has(k.id)),
          ]

          return {
            ...agent,
            knowledge: mergedKnowledge,
          }
        })
        .concat(
          // Add imported agents that don't exist yet
          importedAgents.filter((agent) => !existingAgentsMap.has(agent.id)),
        )

    default:
      return existingAgents
  }
}
