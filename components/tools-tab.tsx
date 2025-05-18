// Create a new ToolsTab component
"use client"

import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Agent, AgentToolPermissions } from "@/lib/types"

interface ToolsTabProps {
  selectedAgent: Agent
  toolPermissions: AgentToolPermissions
  hasChanges: boolean
  onToolChange: (toolKey: keyof AgentToolPermissions, checked: boolean) => void
}

export default function ToolsTab({ selectedAgent, toolPermissions, hasChanges, onToolChange }: ToolsTabProps) {
  return (
    <ScrollArea className="max-h-[calc(100vh-250px)]" type="always">
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground mb-4">
          Enable or disable tools that this agent can use during conversations.
        </p>

        <div className="space-y-3">
          <h3 className="text-sm font-medium border-b pb-1">Information Access Tools</h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-internet-search"
                checked={toolPermissions.internetSearch}
                onCheckedChange={(checked) => onToolChange("internetSearch", !!checked)}
              />
              <label
                htmlFor="tool-internet-search"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Internet Search
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-web-scraping"
                checked={toolPermissions.webScraping}
                onCheckedChange={(checked) => onToolChange("webScraping", !!checked)}
              />
              <label
                htmlFor="tool-web-scraping"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Web Scraping
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-wikipedia"
                checked={toolPermissions.wikipediaAccess}
                onCheckedChange={(checked) => onToolChange("wikipediaAccess", !!checked)}
              />
              <label
                htmlFor="tool-wikipedia"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Wikipedia Access
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-news-api"
                checked={toolPermissions.newsApi}
                onCheckedChange={(checked) => onToolChange("newsApi", !!checked)}
              />
              <label
                htmlFor="tool-news-api"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                News API
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-academic-search"
                checked={toolPermissions.academicSearch}
                onCheckedChange={(checked) => onToolChange("academicSearch", !!checked)}
              />
              <label
                htmlFor="tool-academic-search"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Academic Paper Search
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-document-retrieval"
                checked={toolPermissions.documentRetrieval}
                onCheckedChange={(checked) => onToolChange("documentRetrieval", !!checked)}
              />
              <label
                htmlFor="tool-document-retrieval"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Document Retrieval
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-medium border-b pb-1">Content Generation & Processing</h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-image-generation"
                checked={toolPermissions.imageGeneration}
                onCheckedChange={(checked) => onToolChange("imageGeneration", !!checked)}
              />
              <label
                htmlFor="tool-image-generation"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Image Generation
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-text-summarization"
                checked={toolPermissions.textSummarization}
                onCheckedChange={(checked) => onToolChange("textSummarization", !!checked)}
              />
              <label
                htmlFor="tool-text-summarization"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Text Summarization
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-translation"
                checked={toolPermissions.translation}
                onCheckedChange={(checked) => onToolChange("translation", !!checked)}
              />
              <label
                htmlFor="tool-translation"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Translation
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-code-execution"
                checked={toolPermissions.codeExecution}
                onCheckedChange={(checked) => onToolChange("codeExecution", !!checked)}
              />
              <label
                htmlFor="tool-code-execution"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Code Execution
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-medium border-b pb-1">Knowledge & Reasoning Tools</h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-calculator"
                checked={toolPermissions.calculator}
                onCheckedChange={(checked) => onToolChange("calculator", !!checked)}
              />
              <label
                htmlFor="tool-calculator"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Calculator
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-knowledge-graph"
                checked={toolPermissions.knowledgeGraphQuery}
                onCheckedChange={(checked) => onToolChange("knowledgeGraphQuery", !!checked)}
              />
              <label
                htmlFor="tool-knowledge-graph"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Knowledge Graph Query
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-fact-checking"
                checked={toolPermissions.factChecking}
                onCheckedChange={(checked) => onToolChange("factChecking", !!checked)}
              />
              <label
                htmlFor="tool-fact-checking"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Fact-Checking
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-timeline"
                checked={toolPermissions.timelineGenerator}
                onCheckedChange={(checked) => onToolChange("timelineGenerator", !!checked)}
              />
              <label
                htmlFor="tool-timeline"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Timeline Generator
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-medium border-b pb-1">External Integrations</h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-weather"
                checked={toolPermissions.weatherData}
                onCheckedChange={(checked) => onToolChange("weatherData", !!checked)}
              />
              <label
                htmlFor="tool-weather"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Weather Data
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-map-location"
                checked={toolPermissions.mapLocationData}
                onCheckedChange={(checked) => onToolChange("mapLocationData", !!checked)}
              />
              <label
                htmlFor="tool-map-location"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Map/Location Data
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-financial-data"
                checked={toolPermissions.financialData}
                onCheckedChange={(checked) => onToolChange("financialData", !!checked)}
              />
              <label
                htmlFor="tool-financial-data"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Financial Data
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-public-datasets"
                checked={toolPermissions.publicDatasets}
                onCheckedChange={(checked) => onToolChange("publicDatasets", !!checked)}
              />
              <label
                htmlFor="tool-public-datasets"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Public Datasets
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-medium border-b pb-1">Agent-Specific Tools</h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-memory-search"
                checked={toolPermissions.memorySearch}
                onCheckedChange={(checked) => onToolChange("memorySearch", !!checked)}
              />
              <label
                htmlFor="tool-memory-search"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Memory Search
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-cross-agent-knowledge"
                checked={toolPermissions.crossAgentKnowledge}
                onCheckedChange={(checked) => onToolChange("crossAgentKnowledge", !!checked)}
              />
              <label
                htmlFor="tool-cross-agent-knowledge"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Cross-Agent Knowledge
              </label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="tool-conversation-analysis"
                checked={toolPermissions.conversationAnalysis}
                onCheckedChange={(checked) => onToolChange("conversationAnalysis", !!checked)}
              />
              <label
                htmlFor="tool-conversation-analysis"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Conversation Analysis
              </label>
            </div>
          </div>
        </div>
      </div>
    </ScrollArea>
  )
}
