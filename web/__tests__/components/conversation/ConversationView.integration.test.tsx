import React from "react";
import { render, screen } from "@testing-library/react";
import { ConversationView } from "@/components/conversation-view";

// Mock the Card component
jest.mock("@/components/ui/card", () => ({
  Card: ({ children, className }: any) => (
    <div className={className} data-testid="card">
      {children}
    </div>
  ),
}));

describe("ConversationView Integration Tests", () => {
  describe("Component Rendering", () => {
    it("renders with conversation ID and live status", () => {
      render(<ConversationView conversationId="conv-123" isLive={true} />);

      expect(screen.getByTestId("card")).toBeInTheDocument();
      expect(screen.getByText("Conversation conv-123")).toBeInTheDocument();
      expect(
        screen.getByText(/Live conversation interface coming soon/),
      ).toBeInTheDocument();
    });

    it("renders with recorded status", () => {
      render(<ConversationView conversationId="conv-456" isLive={false} />);

      expect(screen.getByTestId("card")).toBeInTheDocument();
      expect(screen.getByText("Conversation conv-456")).toBeInTheDocument();
      expect(
        screen.getByText(/Recorded conversation interface coming soon/),
      ).toBeInTheDocument();
    });

    it("applies correct CSS classes", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const card = screen.getByTestId("card");
      expect(card).toHaveClass("w-full", "h-full", "p-6");
    });

    it("renders heading with correct styling", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const heading = screen.getByRole("heading", { level: 2 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveClass("text-2xl", "font-semibold", "mb-4");
      expect(heading).toHaveTextContent("Conversation test");
    });

    it("renders description text with correct styling", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const description = screen.getByText(
        /conversation interface coming soon/,
      );
      expect(description).toHaveClass("text-muted-foreground");
    });
  });

  describe("Prop Handling", () => {
    it("handles different conversation ID formats", () => {
      const testCases = [
        "simple-id",
        "complex-id-with-dashes",
        "id_with_underscores",
        "IdWithCamelCase",
        "123-numeric-start",
        "very-long-conversation-id-with-many-parts",
        "",
      ];

      testCases.forEach((id, index) => {
        const { unmount } = render(
          <ConversationView conversationId={id} isLive={true} />,
        );

        if (id === "") {
          expect(screen.getByText(/^Conversation\s*$/)).toBeInTheDocument();
        } else {
          expect(screen.getByText(`Conversation ${id}`)).toBeInTheDocument();
        }

        unmount();
      });
    });

    it("handles boolean isLive prop correctly", () => {
      // Test true
      const { rerender } = render(
        <ConversationView conversationId="test" isLive={true} />,
      );
      expect(screen.getByText(/Live conversation/)).toBeInTheDocument();

      // Test false
      rerender(<ConversationView conversationId="test" isLive={false} />);
      expect(screen.getByText(/Recorded conversation/)).toBeInTheDocument();
    });

    it("handles special characters in conversation ID", () => {
      const specialIds = [
        "conv-123!@#",
        "conversation with spaces",
        "conv<>quotes",
        "conv&amp;entity",
      ];

      specialIds.forEach((id) => {
        const { unmount } = render(
          <ConversationView conversationId={id} isLive={true} />,
        );

        expect(screen.getByText(`Conversation ${id}`)).toBeInTheDocument();
        unmount();
      });
    });
  });

  describe("State Display Logic", () => {
    it("displays correct text for live conversation", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const statusText = screen.getByText(
        /Live conversation interface coming soon/,
      );
      expect(statusText).toBeInTheDocument();
      expect(statusText).toHaveTextContent(
        "Live conversation interface coming soon...",
      );
    });

    it("displays correct text for recorded conversation", () => {
      render(<ConversationView conversationId="test" isLive={false} />);

      const statusText = screen.getByText(
        /Recorded conversation interface coming soon/,
      );
      expect(statusText).toBeInTheDocument();
      expect(statusText).toHaveTextContent(
        "Recorded conversation interface coming soon...",
      );
    });

    it("conditionally renders status based on isLive prop", () => {
      const { rerender } = render(
        <ConversationView conversationId="test" isLive={true} />,
      );

      expect(screen.getByText(/Live/)).toBeInTheDocument();
      expect(screen.queryByText(/Recorded/)).not.toBeInTheDocument();

      rerender(<ConversationView conversationId="test" isLive={false} />);

      expect(screen.getByText(/Recorded/)).toBeInTheDocument();
      expect(screen.queryByText(/Live/)).not.toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("has proper semantic structure", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      expect(screen.getByRole("heading", { level: 2 })).toBeInTheDocument();
      expect(screen.getByTestId("card")).toBeInTheDocument();
    });

    it("provides meaningful heading text", () => {
      render(
        <ConversationView conversationId="accessible-test" isLive={true} />,
      );

      const heading = screen.getByRole("heading");
      expect(heading).toHaveAccessibleName("Conversation accessible-test");
    });

    it("has descriptive content text", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const description = screen.getByText(
        /Live conversation interface coming soon/,
      );
      expect(description).toBeInTheDocument();
    });

    it("maintains semantic hierarchy", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      // Ensure there's only one h2 element
      const headings = screen.getAllByRole("heading");
      expect(headings).toHaveLength(1);
      expect(headings[0].tagName).toBe("H2");
    });
  });

  describe("Component Structure", () => {
    it("wraps content in Card component", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const card = screen.getByTestId("card");
      expect(card).toBeInTheDocument();

      // Check that heading and description are inside the card
      expect(card).toContainElement(screen.getByRole("heading"));
      expect(card).toContainElement(
        screen.getByText(/conversation interface coming soon/),
      );
    });

    it("maintains consistent layout structure", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const card = screen.getByTestId("card");
      const heading = screen.getByRole("heading");
      const description = screen.getByText(
        /conversation interface coming soon/,
      );

      // Verify parent-child relationships
      expect(card).toContainElement(heading);
      expect(card).toContainElement(description);

      // Verify order (heading should come before description in DOM)
      const cardChildren = Array.from(card.children);
      const headingIndex = cardChildren.indexOf(heading);
      const descriptionIndex = cardChildren.indexOf(description);
      expect(headingIndex).toBeLessThan(descriptionIndex);
    });
  });

  describe("Edge Cases", () => {
    it("handles undefined conversationId gracefully", () => {
      // TypeScript would normally prevent this, but test runtime behavior
      expect(() => {
        render(
          <ConversationView conversationId={undefined as any} isLive={true} />,
        );
      }).not.toThrow();
    });

    it("handles null conversationId gracefully", () => {
      // TypeScript would normally prevent this, but test runtime behavior
      expect(() => {
        render(<ConversationView conversationId={null as any} isLive={true} />);
      }).not.toThrow();
    });

    it("handles undefined isLive gracefully", () => {
      expect(() => {
        render(
          <ConversationView conversationId="test" isLive={undefined as any} />,
        );
      }).not.toThrow();
    });

    it("handles very long conversation IDs", () => {
      const longId = "a".repeat(1000);
      render(<ConversationView conversationId={longId} isLive={true} />);

      expect(screen.getByText(`Conversation ${longId}`)).toBeInTheDocument();
    });

    it("handles empty string conversation ID", () => {
      render(<ConversationView conversationId="" isLive={true} />);

      expect(screen.getByText(/^Conversation\s*$/)).toBeInTheDocument();
    });
  });

  describe("Performance", () => {
    it("renders quickly with standard props", () => {
      const startTime = performance.now();
      render(<ConversationView conversationId="perf-test" isLive={true} />);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should render in under 100ms
    });

    it("handles rapid prop changes efficiently", () => {
      const { rerender } = render(
        <ConversationView conversationId="test-1" isLive={true} />,
      );

      // Rapidly change props
      for (let i = 0; i < 10; i++) {
        rerender(
          <ConversationView
            conversationId={`test-${i}`}
            isLive={i % 2 === 0}
          />,
        );
      }

      // Should end up with the final state
      expect(screen.getByText("Conversation test-9")).toBeInTheDocument();
      expect(screen.getByText(/Recorded conversation/)).toBeInTheDocument();
    });
  });

  describe("Component Lifecycle", () => {
    it("mounts without errors", () => {
      expect(() => {
        render(<ConversationView conversationId="mount-test" isLive={true} />);
      }).not.toThrow();
    });

    it("unmounts without errors", () => {
      const { unmount } = render(
        <ConversationView conversationId="unmount-test" isLive={true} />,
      );

      expect(() => unmount()).not.toThrow();
    });

    it("handles multiple mounts and unmounts", () => {
      for (let i = 0; i < 5; i++) {
        const { unmount } = render(
          <ConversationView conversationId={`lifecycle-${i}`} isLive={true} />,
        );

        expect(
          screen.getByText(`Conversation lifecycle-${i}`),
        ).toBeInTheDocument();
        unmount();
      }
    });

    it("preserves prop changes across re-renders", () => {
      const { rerender } = render(
        <ConversationView conversationId="original" isLive={true} />,
      );

      expect(screen.getByText("Conversation original")).toBeInTheDocument();
      expect(screen.getByText(/Live/)).toBeInTheDocument();

      rerender(<ConversationView conversationId="updated" isLive={false} />);

      expect(screen.getByText("Conversation updated")).toBeInTheDocument();
      expect(screen.getByText(/Recorded/)).toBeInTheDocument();
      expect(
        screen.queryByText("Conversation original"),
      ).not.toBeInTheDocument();
    });
  });

  describe("Text Content Validation", () => {
    it("formats heading text correctly", () => {
      render(<ConversationView conversationId="format-test" isLive={true} />);

      const heading = screen.getByRole("heading");
      expect(heading.textContent).toBe("Conversation format-test");
      expect(heading.textContent).toMatch(/^Conversation /);
    });

    it("includes conversation ID in heading", () => {
      const testId = "specific-conversation-id";
      render(<ConversationView conversationId={testId} isLive={true} />);

      const heading = screen.getByRole("heading");
      expect(heading.textContent).toContain(testId);
    });

    it("displays complete status message", () => {
      render(<ConversationView conversationId="test" isLive={true} />);

      const statusMessage = screen.getByText(
        /Live conversation interface coming soon/,
      );
      expect(statusMessage.textContent).toBe(
        "Live conversation interface coming soon...",
      );
    });

    it("maintains text consistency across re-renders", () => {
      const { rerender } = render(
        <ConversationView conversationId="consistency" isLive={true} />,
      );

      let statusText = screen.getByText(
        /Live conversation interface coming soon/,
      );
      expect(statusText.textContent).toBe(
        "Live conversation interface coming soon...",
      );

      rerender(
        <ConversationView conversationId="consistency" isLive={false} />,
      );

      statusText = screen.getByText(
        /Recorded conversation interface coming soon/,
      );
      expect(statusText.textContent).toBe(
        "Recorded conversation interface coming soon...",
      );
    });
  });
});
