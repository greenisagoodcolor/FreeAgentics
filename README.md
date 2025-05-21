This work, "CogniticNet," by Andrew Blake Pashea, is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/

# CogniticNet

CogniticNet is a multi-agent UI design grid world for creating, managing, and observing autonomous AI agent interactions. This experimental platform enables researchers and developers to explore emergent behaviors in multi-agent systems through a visual, interactive interface.

![CogniticNet Screenshot 002](https://drive.google.com/uc?id=18B1O0jUQz3cIcahUrceg7ajlxko6iWKB)

## Features

- **Multi-Agent Environment**: Create and manage multiple AI agents with unique characteristics
- **Grid World Visualization**: Position agents in a 2D grid to trigger proximity-based interactions
- **Autonomous Conversations**: Agents can initiate and participate in conversations based on various triggers
- **Knowledge Management**: Add, edit, and extract beliefs from conversations to build agent knowledge bases
- **Global Knowledge Graph**: Visualize connections between agents and their knowledge
- **Secure API Key Management**: Safely store and manage LLM API keys
- **Import/Export Functionality**: Save and load agent configurations and conversations

## File Structure

```text
CogniticNet/
├── app/                    # Next.js app directory
│   ├── api/                # API routes for secure operations
│   │   ├── api-key/        # API key management endpoints
│   │   └── llm/            # LLM integration endpoints
│   ├── layout.tsx          # Main app layout
│   ├── page.tsx            # Main application page
│   └── settings/           # Settings page
├── components/             # React components
│   ├── agent-list.tsx      # Agent management interface
│   ├── autonomous-conversation-manager.tsx  # Manages agent conversations
│   ├── chat-window.tsx     # Conversation interface
│   ├── global-knowledge-graph.tsx  # Knowledge visualization
│   ├── grid-world.tsx      # 2D grid environment
│   ├── memory-viewer.tsx   # Agent memory and knowledge interface
│   └── ui/                 # UI components (buttons, cards, etc.)
├── contexts/               # React contexts
│   ├── is-sending-context.tsx  # Message sending state
│   └── llm-context.tsx     # LLM client context
├── hooks/                  # Custom React hooks
│   ├── use-autonomous-conversations.ts  # Hook for autonomous conversations
│   └── use-conversation-orchestrator.ts  # Conversation management
├── lib/                    # Core functionality
│   ├── autonomous-conversation.ts  # Autonomous conversation logic
│   ├── belief-extraction.ts  # Extract beliefs from conversations
│   ├── conversation-dynamics.ts  # Conversation flow management
│   ├── encryption.ts       # Secure data handling
│   ├── knowledge-export.ts  # Export functionality
│   ├── knowledge-import.ts  # Import functionality
│   ├── llm-service.ts      # LLM integration
│   ├── llm-settings.ts     # LLM configuration
│   └── types.ts            # TypeScript type definitions
```

## Setup Instructions

### Prerequisites

- Node.js 18.x or later
- npm or yarn
- An API key from an LLM provider (OpenAI, Anthropic, etc.)

### Installation

1. Clone the repository:
   \\\bash
   git clone https://github.com/apashea/CogniticNet.git
   cd CogniticNet
   \\\

2. Install dependencies:
   \\\bash
   npm install
   # or
   yarn install
   \\\

3. Create a `.env.local` file in the root directory with your encryption key:
   \\\
   ENCRYPTION_KEY=your-encryption-key-here
   \\\

4. Start the development server:
   \\\bash
   npm run dev
   # or
   yarn dev
   \\\

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Configuration

1. When you first open the application, you'll be prompted to enter your LLM API key.
2. Select your preferred LLM provider and model.
3. Configure autonomous conversation settings as needed.

## Usage

1. **Create Agents**: Use the agent list panel to create new agents with unique names and biographies.
2. **Position Agents**: Drag agents in the grid world to position them.
3. **Knowledge Management**: Add knowledge entries to agents through the memory viewer.
4. **Observe Interactions**: Watch as agents autonomously interact based on proximity and other triggers.
5. **Export/Import**: Save your work using the export functionality and reload it later.

### Import/Export Functionality

CogniticNet allows you to save and restore your entire workspace, including agents, knowledge, and settings:

#### Exporting Your Work

1. Click the "Export" button in the agent list panel
2. Select which agents to export
3. Choose export options:
   - **Include Settings**: Exports your LLM provider and model settings
   - **Include API Keys**: Includes your API keys in the export (recommended for personal backups)
4. Click "Export" to download a ZIP file containing your agents and settings

#### Importing Your Work

1. Click the "Import" button in the agent list panel
2. Select the ZIP file you previously exported
3. Choose import options:
   - **Import Settings**: Restores your LLM provider and model settings
   - **Import API Keys**: Restores your API keys (only if they were included in the export)
4. Click "Import" to restore your agents and settings

**Note**: Imported API keys are securely stored and will persist across browser sessions, allowing you to continue your work seamlessly after importing.

## License

Dual Licensed:

- **Creative Commons CC BY-NC-SA** for non-commercial use
  - BY - Attribution: Credit must be given to the original creator
  - NC - NonCommercial: May not be used for commercial purposes
  - SA - ShareAlike: Adaptations must be shared under the same terms
- **Commercial License** available for proprietary applications

## Disclaimer

CogniticNet is currently in early development (version 0.0.1) and is provided for research and experimental purposes only. The information, knowledge, and interactions generated by agents within this application may be incomplete, inaccurate, or misleading. Users should not rely on any outputs from this system for critical decisions or as factual truth.

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests or issues. All contributions require a Contributor License Agreement (CLA).
