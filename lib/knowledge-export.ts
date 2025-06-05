import JSZip from "jszip"
import saveAs from "file-saver" // Fixed: Changed from import { saveAs } to import saveAs
import type { Agent, Conversation } from "./types"
import type { LLMSettings } from "./llm-settings"
import { createLogger } from "./debug-logger"
import { getApiKey } from "./api-key-storage"

// Create a module-specific logger
const logger = createLogger("knowledge-export")

interface ExportOptions {
  includeSettings?: boolean
  settings?: LLMSettings
  includeApiKeys?: boolean
  includeConversations?: boolean
  conversations?: Conversation[]
}

/**
 * Convert a knowledge entry to markdown format
 * @param entry Knowledge entry to convert
 * @returns Markdown string representation
 */
export function knowledgeToMarkdown(entry: any): string {
  // Create YAML frontmatter with metadata
  const frontmatter = [
    "---",
    `id: ${entry.id}`,
    `title: ${entry.title}`,
    `tags: ${entry.tags.join(", ")}`,
    `timestamp: ${entry.timestamp.toISOString()}`,
    "---",
    "",
  ].join("\n")

  // Combine frontmatter with content
  return `${frontmatter}${entry.content}`
}

/**
 * Sanitizes a string to be used as a filename
 */
export function sanitizeFilename(name: string): string {
  // Replace invalid filename characters with underscores
  return name
    .replace(/[/\\?%*:|"<>]/g, "_")
    .replace(/\s+/g, "_")
    .toLowerCase()
}

/**
 * Exports an agent's knowledge to a ZIP file
 */
export async function exportAgentKnowledge(agent: Agent): Promise<void> {
  try {
    // Create a new ZIP file
    const zip = new JSZip()

    // Create a folder for the agent
    const agentFolder = zip.folder(sanitizeFilename(agent.name))
    if (!agentFolder) throw new Error("Failed to create agent folder")

    // Add agent metadata
    const agentMetadata = {
      id: agent.id,
      name: agent.name,
      biography: agent.biography,
      color: agent.color,
      autonomyEnabled: agent.autonomyEnabled,
      // We don't include position as it's transient
      // We don't include inConversation as it's a runtime state
    }
    agentFolder.file("_agent.json", JSON.stringify(agentMetadata, null, 2))

    // Create a knowledge folder
    const knowledgeFolder = agentFolder.folder("knowledge")
    if (!knowledgeFolder) throw new Error("Failed to create knowledge folder")

    // Add each knowledge entry as a markdown file
    for (const entry of agent.knowledge) {
      const filename = `${sanitizeFilename(entry.title)}.md`
      const content = knowledgeToMarkdown(entry)
      knowledgeFolder.file(filename, content)
    }

    // Generate the ZIP file
    const content = await zip.generateAsync({ type: "blob" })

    // Trigger download
    saveAs(content, `${sanitizeFilename(agent.name)}_knowledge.zip`)

    return Promise.resolve()
  } catch (error) {
    console.error("Error exporting agent knowledge:", error)
    return Promise.reject(error)
  }
}

/**
 * Export agents' knowledge to a ZIP file
 * @param agents Agents to export
 * @param options Export options
 */
export async function exportAgentsKnowledge(agents: Agent[], options: ExportOptions = {}) {
  try {
    const zip = new JSZip()
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
    const filename = `agents-export-${timestamp}.zip`

    // Add each agent to the zip
    agents.forEach((agent) => {
      // Create a sanitized version of the agent for export
      const exportAgent = { ...agent }

      // Add the agent data to the zip
      zip.file(`agents/${agent.id}.json`, JSON.stringify(exportAgent, null, 2))

      // Add each knowledge entry as a separate markdown file
      agent.knowledge.forEach((entry) => {
        const entryFilename = `${entry.id}.md`
        const entryContent = `# ${entry.title}

${entry.content}`
        zip.file(`knowledge/${agent.id}/${entryFilename}`, entryContent)
      })
    })

    // Add settings if requested
    if (options.includeSettings && options.settings) {
      // Create a sanitized version of the settings
      const exportSettings = { ...options.settings }

      // Handle API key if requested
      if (options.includeApiKeys && exportSettings.apiKeySessionId) {
        try {
          // Retrieve the actual API key for export
          const apiKey = await getApiKey(exportSettings.provider, exportSettings.apiKeySessionId)
          if (apiKey) {
            // Add the API key to the export settings
            exportSettings.apiKey = apiKey
          } else {
            logger.warn("Could not retrieve API key for export")
          }
        } catch (error) {
          logger.error("Error retrieving API key for export", error)
        }
      }

      // Always remove the session ID from exports as it's only valid for the current browser
      delete exportSettings.apiKeySessionId

      // Add the settings to the zip
      zip.file("settings.json", JSON.stringify(exportSettings, null, 2))
    }

    // Add conversations if requested
    if (options.includeConversations && options.conversations && options.conversations.length > 0) {
      // Create a conversations folder
      const conversationsFolder = zip.folder("conversations")

      // Add each conversation as a JSON file
      options.conversations.forEach((conversation) => {
        // Create a sanitized version of the conversation for export
        const exportConversation = {
          id: conversation.id,
          participants: conversation.participants,
          messages: conversation.messages,
          startTime: conversation.startTime,
          endTime: conversation.endTime,
          isAutonomous: conversation.isAutonomous,
          trigger: conversation.trigger,
          topic: conversation.topic,
        }

        // Add the conversation to the zip
        conversationsFolder?.file(`${conversation.id}.json`, JSON.stringify(exportConversation, null, 2))
      })

      // Also create a markdown summary of conversations
      let conversationSummary = "# Conversation History\n\n"
      options.conversations.forEach((conversation) => {
        const startTime = new Date(conversation.startTime).toLocaleString()
        const endTime = conversation.endTime ? new Date(conversation.endTime).toLocaleString() : "Ongoing"
        const participantIds = conversation.participants.join(", ")
        const messageCount = conversation.messages.length

        conversationSummary += `## Conversation ${conversation.id}\n\n`
        conversationSummary += `- **Start Time:** ${startTime}\n`
        conversationSummary += `- **End Time:** ${endTime}\n`
        conversationSummary += `- **Participants:** ${participantIds}\n`
        conversationSummary += `- **Message Count:** ${messageCount}\n`
        conversationSummary += `- **Type:** ${conversation.isAutonomous ? "Autonomous" : "Manual"}\n`
        if (conversation.trigger) {
          conversationSummary += `- **Trigger:** ${conversation.trigger}\n`
        }
        if (conversation.topic) {
          conversationSummary += `- **Topic:** ${conversation.topic}\n`
        }
        conversationSummary += "\n"
      })

      conversationsFolder?.file("_summary.md", conversationSummary)
    }

    // Generate the zip file
    const content = await zip.generateAsync({ type: "blob" })

    // Save the zip file
    saveAs(content, filename)

    logger.info(`Exported ${agents.length} agents to ${filename}`)
    return filename
  } catch (error) {
    logger.error("Error exporting agents:", error)
    throw new Error(`Failed to export agents: ${error instanceof Error ? error.message : String(error)}`)
  }
}
