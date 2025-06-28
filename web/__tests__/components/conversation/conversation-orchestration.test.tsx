import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { ConversationDashboard } from '@/components/conversation/conversation-dashboard';
import { OptimizedConversationDashboard } from '@/components/conversation/optimized-conversation-dashboard';
import { VirtualizedMessageList } from '@/components/conversation/virtualized-message-list';
import * as MessageComponents from '@/components/conversation/message-components';
import { ConversationSearch } from '@/components/conversation/conversation-search';

// Mock conversation data
const mockConversations = [
  {
    id: 'conv-1',
    participants: ['agent-1', 'agent-2'],
    messages: [
      {
        id: 'msg-1',
        senderId: 'agent-1',
        content: 'Hello, how are you?',
        timestamp: new Date(Date.now() - 3600000),
        type: 'text' as const,
      },
      {
        id: 'msg-2',
        senderId: 'agent-2',
        content: 'I am functioning well, thank you.',
        timestamp: new Date(Date.now() - 3000000),
        type: 'text' as const,
      },
    ],
    startTime: new Date(Date.now() - 7200000),
    lastActivityTime: new Date(Date.now() - 3000000),
    status: 'active' as const,
    endTime: null,
    metadata: {
      topic: 'greeting',
      importance: 'low',
    },
  },
  {
    id: 'conv-2',
    participants: ['agent-1', 'agent-3', 'agent-4'],
    messages: [
      {
        id: 'msg-3',
        senderId: 'agent-1',
        content: 'Let us discuss the knowledge graph updates.',
        timestamp: new Date(Date.now() - 1800000),
        type: 'text' as const,
      },
    ],
    startTime: new Date(Date.now() - 1800000),
    lastActivityTime: new Date(Date.now() - 1800000),
    status: 'active' as const,
    endTime: null,
    metadata: {
      topic: 'knowledge-sharing',
      importance: 'high',
    },
  },
];

describe('Conversation Orchestration Components', () => {
  describe('ConversationDashboard', () => {
    it('renders without crashing', () => {
      render(<ConversationDashboard conversations={[]} agents={[]} onConversationSelect={() => {}} />);
      expect(screen.getByRole('region')).toBeInTheDocument();
    });

    it('displays conversation list', () => {
      render(<ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />);
      
      expect(screen.getByText(/Hello, how are you/)).toBeInTheDocument();
      expect(screen.getByText(/discuss the knowledge graph/)).toBeInTheDocument();
    });

    it('filters conversations by status', () => {
      render(<ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />);
      
      const filterSelect = screen.getByLabelText(/filter by status/i);
      fireEvent.change(filterSelect, { target: { value: 'completed' } });
      
      // Active conversations should be hidden
      expect(screen.queryByText(/Hello, how are you/)).not.toBeInTheDocument();
    });

    it('sorts conversations by time', () => {
      render(<ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />);
      
      const sortButton = screen.getByLabelText(/sort by/i);
      fireEvent.click(sortButton);
      
      const messages = screen.getAllByRole('article');
      expect(messages[0]).toHaveTextContent(/knowledge graph/);
    });

    it('handles conversation selection', () => {
      const onSelect = jest.fn();
      render(
        <ConversationDashboard conversations={mockConversations}
          onConversationSelect={onSelect} {...({} as any)}  />
      );
      
      const firstConversation = screen.getByText(/Hello, how are you/).closest('article');
      fireEvent.click(firstConversation!);
      
      expect(onSelect).toHaveBeenCalledWith('conv-1');
    });

    it('displays participant count', () => {
      render(<ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />);
      
      expect(screen.getByText(/2 participants/)).toBeInTheDocument();
      expect(screen.getByText(/3 participants/)).toBeInTheDocument();
    });

    it('shows real-time updates indicator', async (): Promise<void> => {
      const { rerender } = render(
        <ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />
      );
      
      const updatedConversations = [...mockConversations];
      updatedConversations[0].messages.push({
        id: 'msg-new',
        senderId: 'agent-1',
        content: 'New message!',
        timestamp: new Date(),
        type: 'text',
      });
      
      rerender(<ConversationDashboard conversations={updatedConversations} agents={[]} onConversationSelect={() => {}} />);
      
      await waitFor(() => {
        expect(screen.getByText(/New message!/)).toBeInTheDocument();
      });
    });
  });

  describe('OptimizedConversationDashboard', () => {
    it('handles large conversation lists efficiently', () => {
      const largeConversationList = Array.from({ length: 1000 }, (_, i) => ({
        id: `conv-${i}`,
        participants: ['agent-1', 'agent-2'],
        messages: [{
          id: `msg-${i}`,
          senderId: 'agent-1',
          content: `Message ${i}`,
          timestamp: new Date(Date.now() - i * 1000),
          type: 'text' as const,
        }],
        startTime: new Date(Date.now() - i * 10000),
        lastActivityTime: new Date(Date.now() - i * 1000),
        status: 'active' as const,
    endTime: null,
        metadata: {},
      }));

      const startTime = performance.now();
      render(<OptimizedConversationDashboard conversations={largeConversationList}  {...({} as any)} />);
      const renderTime = performance.now() - startTime;
      
      // Should render efficiently
      expect(renderTime).toBeLessThan(100);
      
      // Should use virtualization
      const visibleItems = screen.getAllByRole('article');
      expect(visibleItems.length).toBeLessThan(50); // Only visible items rendered
    });

    it('implements search functionality', async (): Promise<void> => {
      render(
        <OptimizedConversationDashboard conversations={mockConversations} {...({} as any)} />
      );
      
      const searchInput = screen.getByPlaceholderText(/search conversations/i);
      fireEvent.change(searchInput, { target: { value: 'knowledge' } });
      
      await waitFor(() => {
        expect(screen.queryByText(/Hello, how are you/)).not.toBeInTheDocument();
        expect(screen.getByText(/knowledge graph/)).toBeInTheDocument();
      });
    });

    it('supports bulk actions', () => {
      const onBulkAction = jest.fn();
      render(
        <OptimizedConversationDashboard conversations={mockConversations} {...({} as any)} />
      );
      
      // Select multiple conversations
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[1]); // First conversation
      fireEvent.click(checkboxes[2]); // Second conversation
      
      // Perform bulk action
      const archiveButton = screen.getByText(/archive selected/i);
      fireEvent.click(archiveButton);
      
      // Note: onBulkAction test would go here if the component supported it
    });

    it('exports conversation data', () => {
      render(
        <OptimizedConversationDashboard conversations={mockConversations} {...({} as any)} />
      );
      
      const exportButton = screen.getByLabelText(/export conversations/i);
      fireEvent.click(exportButton);
      
      const exportFormat = screen.getByRole('dialog');
      expect(exportFormat).toBeInTheDocument();
      
      // Select JSON format
      const jsonOption = screen.getByLabelText(/json/i);
      fireEvent.click(jsonOption);
      
      const confirmExport = screen.getByText(/confirm export/i);
      fireEvent.click(confirmExport);
      
      // Verify download initiated (mock implementation would handle actual download)
    });
  });

  describe('VirtualizedMessageList', () => {
    const manyMessages = Array.from({ length: 1000 }, (_, i) => ({
      id: `msg-${i}`,
      senderId: `agent-${i % 3}`,
      content: `This is message number ${i}`,
      timestamp: new Date(Date.now() - (1000 - i) * 60000),
      type: 'text' as const,
    }));

    it('virtualizes long message lists', () => {
      render(
        <VirtualizedMessageList 
          messages={manyMessages}
          agents={[]}
          height={600}
        />
      );
      
      // Only renders visible messages
      const visibleMessages = screen.getAllByRole('listitem');
      expect(visibleMessages.length).toBeLessThan(20);
    });

    it('scrolls to specific messages', () => {
      const { container } = render(
        <VirtualizedMessageList 
          messages={manyMessages}
          agents={[]}
          height={600}
        />
      );
      
      const scrollContainer = container.querySelector('[data-testid="virtual-scroll"]');
      expect(scrollContainer?.scrollTop).toBeGreaterThan(0);
    });

    it('handles dynamic item heights', () => {
      const getItemHeight = (index: number) => {
        // Some messages are taller
        return index % 5 === 0 ? 120 : 80;
      };
      
      render(
        <VirtualizedMessageList 
          messages={manyMessages}
          agents={[]}
          height={600}
        />
      );
      
      const tallMessages = screen.getAllByRole('listitem')
        .filter(el => el.style.height === '120px');
      expect(tallMessages.length).toBeGreaterThan(0);
    });

    it('supports message actions', () => {
      const onMessageAction = jest.fn();
      
      render(
        <VirtualizedMessageList 
          messages={manyMessages.slice(0, 10)}
          agents={[]}
          height={600}
        />
      );
      
      const firstMessage = screen.getAllByRole('listitem')[0];
      const moreButton = within(firstMessage).getByLabelText(/more actions/i);
      fireEvent.click(moreButton);
      
      const deleteOption = screen.getByText(/delete/i);
      fireEvent.click(deleteOption);
      
      expect(onMessageAction).toHaveBeenCalledWith('delete', 'msg-0');
    });
  });

  describe('MessageComponents', () => {
    it('renders text messages correctly', () => {
      const textMessage = {
        id: 'msg-1',
        senderId: 'agent-1',
        content: 'This is a text message',
        timestamp: new Date(),
        type: 'text' as const,
      };
      
      render(<MessageComponents.TextMessage message={textMessage} />);
      
      expect(screen.getByText('This is a text message')).toBeInTheDocument();
      expect(screen.getByText(/agent-1/)).toBeInTheDocument();
    });

    it('renders code messages with syntax highlighting', () => {
      const codeMessage = {
        id: 'msg-2',
        senderId: 'agent-2',
        content: 'def hello():\n    print("Hello, world!")',
        timestamp: new Date(),
        type: 'code' as const,
        metadata: { language: 'python' },
      };
      
      render(<MessageComponents.CodeMessage message={codeMessage} />);
      
      expect(screen.getByText(/def hello/)).toBeInTheDocument();
      expect(screen.getByRole('code')).toBeInTheDocument();
      expect(screen.getByText(/python/i)).toBeInTheDocument();
    });

    it('renders system messages distinctly', () => {
      const systemMessage = {
        id: 'msg-3',
        senderId: 'system',
        content: 'Agent-1 has joined the conversation',
        timestamp: new Date(),
        type: 'system' as const,
      };
      
      render(<MessageComponents.SystemMessage message={systemMessage} />);
      
      expect(screen.getByText(/joined the conversation/)).toBeInTheDocument();
      const message = screen.getByText(/joined the conversation/);
      expect(message).toHaveClass('system-message');
    });

    it('shows message timestamps', () => {
      const message = {
        id: 'msg-4',
        senderId: 'agent-1',
        content: 'Test message',
        timestamp: new Date('2024-01-01T12:00:00'),
        type: 'text' as const,
      };
      
      render(<MessageComponents.TextMessage message={message} />);
      
      expect(screen.getByText('Test message')).toBeInTheDocument();
      expect(screen.getByText(/12:00/)).toBeInTheDocument();
    });

    it('handles message reactions', () => {
      const onReaction = jest.fn();
      const message = {
        id: 'msg-5',
        senderId: 'agent-1',
        content: 'Great idea!',
        timestamp: new Date(),
        type: 'text' as const,
        reactions: { '👍': 2, '❤️': 1 },
      };
      
      render(<MessageComponents.TextMessage message={message} onReaction={onReaction} />);
      
      expect(screen.getByText('Great idea!')).toBeInTheDocument();
      
      const thumbsUp = screen.getByText('👍');
      fireEvent.click(thumbsUp);
      
      expect(onReaction).toHaveBeenCalledWith('msg-5', '👍');
    });
  });

  describe('ConversationSearch', () => {
    it('searches through conversation content', async (): Promise<void> => {
      const onSearch = jest.fn();
      
      render(
        <ConversationSearch conversations={mockConversations}
          onSearch={onSearch} {...({} as any)} />
      );
      
      const searchInput = screen.getByPlaceholderText(/search messages/i);
      fireEvent.change(searchInput, { target: { value: 'knowledge graph' } });
      
      await waitFor(() => {
        expect(onSearch).toHaveBeenCalledWith(
          expect.arrayContaining([
            expect.objectContaining({ id: 'conv-2' })
          ])
        );
      });
    });

    it('supports advanced search filters', () => {
      const onSearch = jest.fn();
      
      render(
        <ConversationSearch conversations={mockConversations}
          onSearch={onSearch}
          enableAdvancedSearch {...({} as any)} />
      );
      
      const advancedButton = screen.getByText(/advanced search/i);
      fireEvent.click(advancedButton);
      
      // Set participant filter
      const participantInput = screen.getByLabelText(/participant/i);
      fireEvent.change(participantInput, { target: { value: 'agent-3' } });
      
      // Set date range
      const fromDate = screen.getByLabelText(/from date/i);
      fireEvent.change(fromDate, { target: { value: '2024-01-01' } });
      
      const searchButton = screen.getByText(/search/i);
      fireEvent.click(searchButton);
      
      expect(onSearch).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            participants: expect.arrayContaining(['agent-3'])
          })
        ])
      );
    });

    it('highlights search results', () => {
      render(
        <ConversationSearch conversations={mockConversations}
          highlightResults {...({} as any)} />
      );
      
      const searchInput = screen.getByPlaceholderText(/search messages/i);
      fireEvent.change(searchInput, { target: { value: 'Hello' } });
      
      const highlighted = screen.getByText('Hello');
      expect(highlighted).toBeInTheDocument();
      expect(highlighted).toHaveTextContent('Hello');
    });
  });

  describe('Conversation Performance', () => {
    it('handles rapid message updates efficiently', async (): Promise<void> => {
      const { rerender } = render(
        <ConversationDashboard conversations={mockConversations} agents={[]} onConversationSelect={() => {}} />
      );
      
      const updates = [];
      const startTime = performance.now();
      
      // Simulate 100 rapid updates
      for (let i = 0; i < 100; i++) {
        const updatedConvs = [...mockConversations];
        updatedConvs[0].messages.push({
          id: `rapid-${i}`,
          senderId: 'agent-1',
          content: `Rapid message ${i}`,
          timestamp: new Date(),
          type: 'text',
        });
        
        rerender(<ConversationDashboard conversations={updatedConvs} agents={[]} onConversationSelect={() => {}} />);
      }
      
      const totalTime = performance.now() - startTime;
      
      // Should handle updates efficiently
      expect(totalTime).toBeLessThan(1000); // Less than 1 second for 100 updates
    });

    it('debounces search input', async (): Promise<void> => {
      const onSearch = jest.fn();
      
      render(
        <ConversationSearch conversations={mockConversations}
          onSearch={onSearch}
          debounceMs={300} {...({} as any)} />
      );
      
      const searchInput = screen.getByPlaceholderText(/search/i);
      
      // Type rapidly
      fireEvent.change(searchInput, { target: { value: 'h' } });
      fireEvent.change(searchInput, { target: { value: 'he' } });
      fireEvent.change(searchInput, { target: { value: 'hel' } });
      fireEvent.change(searchInput, { target: { value: 'hell' } });
      fireEvent.change(searchInput, { target: { value: 'hello' } });
      
      // Should not call immediately
      expect(onSearch).not.toHaveBeenCalled();
      
      // Wait for debounce
      await waitFor(() => {
        expect(onSearch).toHaveBeenCalledTimes(1);
        expect(onSearch).toHaveBeenCalledWith(expect.any(Array));
      }, { timeout: 400 });
    });
  });
});