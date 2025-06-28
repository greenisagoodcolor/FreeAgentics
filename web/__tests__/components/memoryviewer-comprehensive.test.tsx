/**
 * Comprehensive Memory Viewer Tests
 * 
 * Tests for memory viewer component functionality
 * following ADR-007 requirements for complete system coverage.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { jest } from '@jest/globals';
import MemoryViewer, { type AgentToolPermissions } from '@/components/memoryviewer';
import type { Agent, Conversation, KnowledgeEntry } from '@/lib/types';

// Mock dependencies
jest.mock('@/hooks/use-toast');
jest.mock('@/contexts/llm-context');
jest.mock('@/lib/utils');
jest.mock('@/lib/belief-extraction');
jest.mock('@/lib/knowledge-export');
jest.mock('@/lib/debug-logger');

// Mock components
jest.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  ),
}));

jest.mock('@/components/ui/textarea', () => ({
  Textarea: ({ value, onChange, ...props }: any) => (
    <textarea value={value} onChange={onChange} {...props} />
  ),
}));

jest.mock('@/components/ui/card', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <h3 {...props}>{children}</h3>,
}));

jest.mock('@/components/ui/select', () => ({
  Select: ({ children, value, onValueChange }: any) => (
    <select value={value} onChange={(e) => onValueChange?.(e.target.value)}>
      {children}
    </select>
  ),
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ value, children }: any) => <option value={value}>{children}</option>,
  SelectTrigger: ({ children }: any) => <>{children}</>,
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
}));

jest.mock('@/components/ui/scroll-area', () => ({
  ScrollArea: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}));

jest.mock('@/components/ui/input', () => ({
  Input: ({ value, onChange, ...props }: any) => (
    <input value={value} onChange={onChange} {...props} />
  ),
}));

jest.mock('@/components/ui/badge', () => ({
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}));

// Mock imports
import { useToast } from '@/hooks/use-toast';
import { useLLM } from '@/contexts/llm-context';
import { formatTimestamp, extractTagsFromMarkdown } from '@/lib/utils';
import { parseBeliefs, parseRefinedBeliefs } from '@/lib/belief-extraction';
import { exportAgentKnowledge } from '@/lib/knowledge-export';

const mockUseToast = useToast as jest.MockedFunction<typeof useToast>;
const mockUseLLM = useLLM as jest.MockedFunction<typeof useLLM>;
const mockFormatTimestamp = formatTimestamp as jest.MockedFunction<typeof formatTimestamp>;
const mockExtractTagsFromMarkdown = extractTagsFromMarkdown as jest.MockedFunction<typeof extractTagsFromMarkdown>;
const mockParseBeliefs = parseBeliefs as jest.MockedFunction<typeof parseBeliefs>;
const mockParseRefinedBeliefs = parseRefinedBeliefs as jest.MockedFunction<typeof parseRefinedBeliefs>;
const mockExportAgentKnowledge = exportAgentKnowledge as jest.MockedFunction<typeof exportAgentKnowledge>;

// Mock data
const mockKnowledgeEntry: KnowledgeEntry = {
  id: 'knowledge-1',
  title: 'Test Knowledge',
  content: 'This is test knowledge content',
  source: 'user',
  timestamp: new Date(),
  tags: ['test', 'knowledge'],
  metadata: {},
};

const mockToolPermissions: AgentToolPermissions = {
  internetSearch: true,
  webScraping: false,
  wikipediaAccess: true,
  newsApi: false,
  academicSearch: true,
  documentRetrieval: false,
  imageGeneration: false,
  textSummarization: true,
  translation: false,
  codeExecution: false,
  calculator: true,
  knowledgeGraphQuery: false,
  factChecking: true,
  timelineGenerator: false,
  weatherData: false,
  mapLocationData: false,
  financialData: false,
  publicDatasets: false,
  memorySearch: true,
  crossAgentKnowledge: false,
  conversationAnalysis: true,
};

const mockAgent: Agent = {
  id: 'agent-1',
  name: 'Test Agent',
  biography: 'This is a test agent biography',
  color: '#ff0000',
  position: { x: 0, y: 0 },
  knowledge: [mockKnowledgeEntry],
  toolPermissions: mockToolPermissions,
  autonomyEnabled: true,
  inConversation: false,
};

const mockConversation: Conversation = {
  id: 'conv-1',
  participants: ['agent-1'],
  messages: [
    {
      id: 'msg-1',
      conversationId: 'conv-1',
      senderId: 'agent-1',
      content: 'Hello, this is a test message.',
      timestamp: new Date(),
      type: 'text',
      metadata: {},
    },
  ],
  createdAt: new Date(),
  updatedAt: new Date(),
  title: 'Test Conversation',
  metadata: {},
};

const defaultProps = {
  selectedAgent: mockAgent,
  conversationHistory: [mockConversation],
  agents: [mockAgent],
  onAddKnowledge: jest.fn(),
  onUpdateAgent: jest.fn(),
  onDeleteKnowledge: jest.fn(),
  onUpdateKnowledge: jest.fn(),
  selectedKnowledgeNode: null,
  onClearSelectedKnowledgeNode: jest.fn(),
  onSelectAgent: jest.fn(),
};

describe('MemoryViewer', () => {
  let mockToast: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockToast = jest.fn();
    mockUseToast.mockReturnValue({
      toast: mockToast,
      dismiss: jest.fn(),
      toasts: [],
    });

    mockUseLLM.mockReturnValue({
      isProcessing: false,
      setIsProcessing: jest.fn(),
      generateResponse: jest.fn(),
      extractBeliefs: jest.fn(),
      generateKnowledgeEntries: jest.fn(),
      llmClient: null,
    });

    mockFormatTimestamp.mockReturnValue('2023-01-01 12:00:00');
    mockExtractTagsFromMarkdown.mockReturnValue(['tag1', 'tag2']);
    mockParseBeliefs.mockReturnValue([]);
    mockParseRefinedBeliefs.mockReturnValue([]);
    mockExportAgentKnowledge.mockResolvedValue('exported data');
  });

  describe('Component Initialization', () => {
    it('renders without errors', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      expect(screen.getByText('Test Agent')).toBeInTheDocument();
    });

    it('displays agent biography', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      expect(screen.getByDisplayValue('This is a test agent biography')).toBeInTheDocument();
    });

    it('handles no selected agent', () => {
      render(<MemoryViewer {...defaultProps} selectedAgent={null} />);
      
      expect(screen.getByText('Select an agent to view their memory')).toBeInTheDocument();
    });

    it('initializes with default view', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Biography view should be selected by default
      expect(screen.getByDisplayValue('This is a test agent biography')).toBeInTheDocument();
    });
  });

  describe('Biography Management', () => {
    it('displays agent biography in textarea', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      const biographyTextarea = screen.getByDisplayValue('This is a test agent biography');
      expect(biographyTextarea).toBeInTheDocument();
    });

    it('updates biography when typing', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      const biographyTextarea = screen.getByDisplayValue('This is a test agent biography');
      fireEvent.change(biographyTextarea, { target: { value: 'Updated biography' } });
      
      expect(biographyTextarea).toHaveValue('Updated biography');
    });

    it('saves biography changes', async () => {
      const onUpdateAgent = jest.fn();
      render(<MemoryViewer {...defaultProps} onUpdateAgent={onUpdateAgent} />);
      
      const biographyTextarea = screen.getByDisplayValue('This is a test agent biography');
      fireEvent.change(biographyTextarea, { target: { value: 'Updated biography' } });
      
      const saveButton = screen.getByText('Save Biography');
      fireEvent.click(saveButton);
      
      await waitFor(() => {
        expect(onUpdateAgent).toHaveBeenCalledWith('agent-1', {
          biography: 'Updated biography',
        });
      });
    });

    it('shows toast on successful save', async () => {
      render(<MemoryViewer {...defaultProps} />);
      
      const biographyTextarea = screen.getByDisplayValue('This is a test agent biography');
      fireEvent.change(biographyTextarea, { target: { value: 'Updated biography' } });
      
      const saveButton = screen.getByText('Save Biography');
      fireEvent.click(saveButton);
      
      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: 'Success',
          description: 'Biography saved successfully',
        });
      });
    });
  });

  describe('Knowledge Management', () => {
    it('displays knowledge entries', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      expect(screen.getByText('Test Knowledge')).toBeInTheDocument();
    });

    it('allows adding new knowledge', async () => {
      const onAddKnowledge = jest.fn();
      render(<MemoryViewer {...defaultProps} onAddKnowledge={onAddKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Switch to add tab
      const addTab = screen.getByText('Add');
      fireEvent.click(addTab);
      
      // Fill in knowledge form
      const titleInput = screen.getByPlaceholderText('Knowledge title...');
      const contentTextarea = screen.getByPlaceholderText('Knowledge content...');
      
      fireEvent.change(titleInput, { target: { value: 'New Knowledge' } });
      fireEvent.change(contentTextarea, { target: { value: 'New knowledge content' } });
      
      // Save knowledge
      const saveButton = screen.getByText('Save Knowledge');
      fireEvent.click(saveButton);
      
      await waitFor(() => {
        expect(onAddKnowledge).toHaveBeenCalledWith('agent-1', expect.objectContaining({
          title: 'New Knowledge',
          content: 'New knowledge content',
        }));
      });
    });

    it('allows editing existing knowledge', async () => {
      const onUpdateKnowledge = jest.fn();
      render(<MemoryViewer {...defaultProps} onUpdateKnowledge={onUpdateKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Select knowledge entry
      const knowledgeItem = screen.getByText('Test Knowledge');
      fireEvent.click(knowledgeItem);
      
      // Click edit button
      const editButton = screen.getByText('Edit');
      fireEvent.click(editButton);
      
      // Update content
      const contentTextarea = screen.getByDisplayValue('This is test knowledge content');
      fireEvent.change(contentTextarea, { target: { value: 'Updated content' } });
      
      // Save changes
      const saveButton = screen.getByText('Save');
      fireEvent.click(saveButton);
      
      await waitFor(() => {
        expect(onUpdateKnowledge).toHaveBeenCalledWith('agent-1', 'knowledge-1', {
          content: 'Updated content',
        });
      });
    });

    it('allows deleting knowledge', async () => {
      const onDeleteKnowledge = jest.fn();
      render(<MemoryViewer {...defaultProps} onDeleteKnowledge={onDeleteKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Select knowledge entry
      const knowledgeItem = screen.getByText('Test Knowledge');
      fireEvent.click(knowledgeItem);
      
      // Click delete button
      const deleteButton = screen.getByText('Delete');
      fireEvent.click(deleteButton);
      
      // Confirm deletion
      const confirmButton = screen.getByText('Confirm Delete');
      fireEvent.click(confirmButton);
      
      await waitFor(() => {
        expect(onDeleteKnowledge).toHaveBeenCalledWith('agent-1', 'knowledge-1');
      });
    });

    it('searches knowledge entries', () => {
      const agentWithMultipleKnowledge = {
        ...mockAgent,
        knowledge: [
          mockKnowledgeEntry,
          {
            ...mockKnowledgeEntry,
            id: 'knowledge-2',
            title: 'Another Knowledge',
            content: 'Different content',
          },
        ],
      };
      
      render(<MemoryViewer {...defaultProps} selectedAgent={agentWithMultipleKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Search for specific knowledge
      const searchInput = screen.getByPlaceholderText('Search knowledge...');
      fireEvent.change(searchInput, { target: { value: 'Test' } });
      
      expect(screen.getByText('Test Knowledge')).toBeInTheDocument();
      expect(screen.queryByText('Another Knowledge')).not.toBeInTheDocument();
    });

    it('filters knowledge by tags', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Select tag filter
      const tagSelect = screen.getByDisplayValue('All Tags');
      fireEvent.change(tagSelect, { target: { value: 'test' } });
      
      expect(screen.getByText('Test Knowledge')).toBeInTheDocument();
    });
  });

  describe('Tool Permissions', () => {
    it('displays tool permissions', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to tools view
      const toolsTab = screen.getByText('Tools');
      fireEvent.click(toolsTab);
      
      expect(screen.getByText('Information Access Tools')).toBeInTheDocument();
      expect(screen.getByText('Internet Search')).toBeInTheDocument();
    });

    it('allows updating tool permissions', async () => {
      const onUpdateAgent = jest.fn();
      render(<MemoryViewer {...defaultProps} onUpdateAgent={onUpdateAgent} />);
      
      // Switch to tools view
      const toolsTab = screen.getByText('Tools');
      fireEvent.click(toolsTab);
      
      // Toggle a permission
      const internetSearchToggle = screen.getByLabelText('Internet Search');
      fireEvent.click(internetSearchToggle);
      
      // Save changes
      const saveButton = screen.getByText('Save Tool Permissions');
      fireEvent.click(saveButton);
      
      await waitFor(() => {
        expect(onUpdateAgent).toHaveBeenCalledWith('agent-1', {
          toolPermissions: expect.objectContaining({
            internetSearch: false, // Should be toggled off
          }),
        });
      });
    });

    it('shows changes indicator when permissions are modified', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to tools view
      const toolsTab = screen.getByText('Tools');
      fireEvent.click(toolsTab);
      
      // Toggle a permission
      const internetSearchToggle = screen.getByLabelText('Internet Search');
      fireEvent.click(internetSearchToggle);
      
      expect(screen.getByText('Unsaved Changes')).toBeInTheDocument();
    });
  });

  describe('Belief Extraction', () => {
    beforeEach(() => {
      mockUseLLM.mockReturnValue({
        isProcessing: false,
        setIsProcessing: jest.fn(),
        generateResponse: jest.fn(),
        extractBeliefs: jest.fn().mockResolvedValue('extracted beliefs'),
        generateKnowledgeEntries: jest.fn(),
        llmClient: null,
      });
    });

    it('displays belief extraction interface', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to inference view
      const inferenceTab = screen.getByText('Inference');
      fireEvent.click(inferenceTab);
      
      expect(screen.getByText('Extract Beliefs from Conversation')).toBeInTheDocument();
    });

    it('allows selecting conversation for extraction', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to inference view
      const inferenceTab = screen.getByText('Inference');
      fireEvent.click(inferenceTab);
      
      // Select conversation
      const conversationSelect = screen.getByDisplayValue('Select a conversation...');
      fireEvent.change(conversationSelect, { target: { value: 'conv-1' } });
      
      expect(screen.getByDisplayValue('Test Conversation')).toBeInTheDocument();
    });

    it('extracts beliefs from conversation', async () => {
      const { extractBeliefs } = mockUseLLM();
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to inference view
      const inferenceTab = screen.getByText('Inference');
      fireEvent.click(inferenceTab);
      
      // Select conversation
      const conversationSelect = screen.getByDisplayValue('Select a conversation...');
      fireEvent.change(conversationSelect, { target: { value: 'conv-1' } });
      
      // Extract beliefs
      const extractButton = screen.getByText('Extract Beliefs');
      fireEvent.click(extractButton);
      
      await waitFor(() => {
        expect(extractBeliefs).toHaveBeenCalled();
      });
    });

    it('displays extraction progress', async () => {
      mockUseLLM.mockReturnValue({
        isProcessing: true,
        setIsProcessing: jest.fn(),
        generateResponse: jest.fn(),
        extractBeliefs: jest.fn().mockImplementation(() => new Promise(() => {})),
        generateKnowledgeEntries: jest.fn(),
        llmClient: null,
      });

      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to inference view
      const inferenceTab = screen.getByText('Inference');
      fireEvent.click(inferenceTab);
      
      // Select conversation and extract
      const conversationSelect = screen.getByDisplayValue('Select a conversation...');
      fireEvent.change(conversationSelect, { target: { value: 'conv-1' } });
      
      const extractButton = screen.getByText('Extract Beliefs');
      fireEvent.click(extractButton);
      
      expect(screen.getByText('Extracting...')).toBeInTheDocument();
    });
  });

  describe('Agent Selection', () => {
    it('updates when agent changes', () => {
      const newAgent = {
        ...mockAgent,
        id: 'agent-2',
        name: 'New Agent',
        biography: 'New biography',
      };

      const { rerender } = render(<MemoryViewer {...defaultProps} />);
      
      expect(screen.getByDisplayValue('This is a test agent biography')).toBeInTheDocument();
      
      rerender(<MemoryViewer {...defaultProps} selectedAgent={newAgent} />);
      
      expect(screen.getByDisplayValue('New biography')).toBeInTheDocument();
    });

    it('resets state when agent changes', () => {
      const newAgent = {
        ...mockAgent,
        id: 'agent-2',
        name: 'New Agent',
        knowledge: [],
      };

      const { rerender } = render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view and select an entry
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      const knowledgeItem = screen.getByText('Test Knowledge');
      fireEvent.click(knowledgeItem);
      
      // Change agent
      rerender(<MemoryViewer {...defaultProps} selectedAgent={newAgent} />);
      
      // Knowledge selection should be reset
      expect(screen.queryByText('Test Knowledge')).not.toBeInTheDocument();
    });
  });

  describe('Knowledge Node Selection', () => {
    it('handles knowledge node selection from global graph', () => {
      const selectedKnowledgeNode = {
        type: 'entry' as const,
        id: 'knowledge-1',
        title: 'Test Knowledge',
      };

      render(<MemoryViewer {...defaultProps} selectedKnowledgeNode={selectedKnowledgeNode} />);
      
      expect(screen.getByText('Knowledge Node Selection')).toBeInTheDocument();
    });

    it('clears knowledge node selection', () => {
      const onClearSelectedKnowledgeNode = jest.fn();
      const selectedKnowledgeNode = {
        type: 'entry' as const,
        id: 'knowledge-1',
        title: 'Test Knowledge',
      };

      render(<MemoryViewer 
        {...defaultProps} 
        selectedKnowledgeNode={selectedKnowledgeNode}
        onClearSelectedKnowledgeNode={onClearSelectedKnowledgeNode}
      />);
      
      const backButton = screen.getByText('Back');
      fireEvent.click(backButton);
      
      expect(onClearSelectedKnowledgeNode).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('handles missing knowledge gracefully', () => {
      const agentWithoutKnowledge = {
        ...mockAgent,
        knowledge: [],
      };

      render(<MemoryViewer {...defaultProps} selectedAgent={agentWithoutKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      expect(screen.getByText('No knowledge entries found')).toBeInTheDocument();
    });

    it('handles API errors during belief extraction', async () => {
      mockUseLLM.mockReturnValue({
        isProcessing: false,
        setIsProcessing: jest.fn(),
        generateResponse: jest.fn(),
        extractBeliefs: jest.fn().mockRejectedValue(new Error('API Error')),
        generateKnowledgeEntries: jest.fn(),
        llmClient: null,
      });

      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to inference view
      const inferenceTab = screen.getByText('Inference');
      fireEvent.click(inferenceTab);
      
      // Select conversation and try to extract
      const conversationSelect = screen.getByDisplayValue('Select a conversation...');
      fireEvent.change(conversationSelect, { target: { value: 'conv-1' } });
      
      const extractButton = screen.getByText('Extract Beliefs');
      fireEvent.click(extractButton);
      
      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: 'Error',
          description: 'Failed to extract beliefs: API Error',
          variant: 'destructive',
        });
      });
    });

    it('handles invalid tool permissions', () => {
      const agentWithInvalidPermissions = {
        ...mockAgent,
        toolPermissions: null,
      };

      render(<MemoryViewer {...defaultProps} selectedAgent={agentWithInvalidPermissions} />);
      
      // Switch to tools view
      const toolsTab = screen.getByText('Tools');
      fireEvent.click(toolsTab);
      
      // Should use default permissions
      expect(screen.getByText('Information Access Tools')).toBeInTheDocument();
    });
  });

  describe('Data Export', () => {
    it('exports agent knowledge', async () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      const exportButton = screen.getByText('Export Knowledge');
      fireEvent.click(exportButton);
      
      await waitFor(() => {
        expect(mockExportAgentKnowledge).toHaveBeenCalledWith(mockAgent);
      });
    });

    it('shows export success message', async () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      const exportButton = screen.getByText('Export Knowledge');
      fireEvent.click(exportButton);
      
      await waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith({
          title: 'Success',
          description: 'Knowledge exported successfully',
        });
      });
    });
  });

  describe('Performance and Memory', () => {
    it('handles large knowledge datasets efficiently', () => {
      const largeKnowledgeSet = Array.from({ length: 1000 }, (_, i) => ({
        ...mockKnowledgeEntry,
        id: `knowledge-${i}`,
        title: `Knowledge ${i}`,
      }));

      const agentWithLargeKnowledge = {
        ...mockAgent,
        knowledge: largeKnowledgeSet,
      };

      const startTime = Date.now();
      render(<MemoryViewer {...defaultProps} selectedAgent={agentWithLargeKnowledge} />);
      const endTime = Date.now();
      
      expect(endTime - startTime).toBeLessThan(1000);
    });

    it('efficiently filters large knowledge sets', () => {
      const largeKnowledgeSet = Array.from({ length: 1000 }, (_, i) => ({
        ...mockKnowledgeEntry,
        id: `knowledge-${i}`,
        title: `Knowledge ${i}`,
        tags: i % 2 === 0 ? ['even'] : ['odd'],
      }));

      const agentWithLargeKnowledge = {
        ...mockAgent,
        knowledge: largeKnowledgeSet,
      };

      render(<MemoryViewer {...defaultProps} selectedAgent={agentWithLargeKnowledge} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Filter by search
      const searchInput = screen.getByPlaceholderText('Search knowledge...');
      fireEvent.change(searchInput, { target: { value: 'Knowledge 1' } });
      
      // Should complete quickly
      expect(screen.getByText('Knowledge 1')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      expect(screen.getByRole('textbox', { name: /biography/i })).toBeInTheDocument();
    });

    it('supports keyboard navigation', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      const biographyTextarea = screen.getByDisplayValue('This is a test agent biography');
      biographyTextarea.focus();
      
      expect(document.activeElement).toBe(biographyTextarea);
    });

    it('provides proper form labels', () => {
      render(<MemoryViewer {...defaultProps} />);
      
      // Switch to knowledge view
      const knowledgeTab = screen.getByText('Knowledge');
      fireEvent.click(knowledgeTab);
      
      // Switch to add tab
      const addTab = screen.getByText('Add');
      fireEvent.click(addTab);
      
      expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/content/i)).toBeInTheDocument();
    });
  });
});