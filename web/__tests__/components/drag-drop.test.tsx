/**
 * Drag and Drop Component Tests
 * Mouse-only interactions for performance optimization
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { jest } from "@jest/globals";

// Mock DOM methods that might be missing
Object.defineProperty(Element.prototype, "getBoundingClientRect", {
  writable: true,
  value: jest.fn(() => ({
    bottom: 0,
    height: 0,
    left: 0,
    right: 0,
    top: 0,
    width: 0,
    x: 0,
    y: 0,
  })),
});

// Drag and Drop System
interface DragState {
  isDragging: boolean;
  draggedElement: string | null;
  dragOffset: { x: number; y: number };
  dropZones: string[];
  validDropZone: string | null;
}

interface DragDropContextValue {
  dragState: DragState;
  startDrag: (elementId: string, offset: { x: number; y: number }) => void;
  updateDrag: (position: { x: number; y: number }) => void;
  endDrag: (dropZoneId?: string) => void;
  registerDropZone: (id: string, accepts: string[]) => void;
  unregisterDropZone: (id: string) => void;
}

const DragDropContext = React.createContext<DragDropContextValue | null>(null);

const DragDropProvider: React.FC<{
  children: React.ReactNode;
  onDrop?: (item: string, zone: string) => void;
}> = ({ children, onDrop }) => {
  const [dragState, setDragState] = React.useState<DragState>({
    isDragging: false,
    draggedElement: null,
    dragOffset: { x: 0, y: 0 },
    dropZones: [],
    validDropZone: null,
  });

  const dropZoneConfig = React.useRef<Map<string, string[]>>(new Map());

  const startDrag = React.useCallback(
    (elementId: string, offset: { x: number; y: number }) => {
      setDragState((prev) => ({
        ...prev,
        isDragging: true,
        draggedElement: elementId,
        dragOffset: offset,
      }));
    },
    [],
  );

  const updateDrag = React.useCallback((position: { x: number; y: number }) => {
    setDragState((prev) => ({
      ...prev,
      dragOffset: position,
    }));
  }, []);

  const endDrag = React.useCallback(
    (dropZoneId?: string) => {
      if (dragState.draggedElement && dropZoneId && onDrop) {
        const accepts = dropZoneConfig.current.get(dropZoneId) || [];
        if (accepts.includes(dragState.draggedElement)) {
          onDrop(dragState.draggedElement, dropZoneId);
        }
      }

      setDragState({
        isDragging: false,
        draggedElement: null,
        dragOffset: { x: 0, y: 0 },
        dropZones: [],
        validDropZone: null,
      });
    },
    [dragState.draggedElement, onDrop],
  );

  const registerDropZone = React.useCallback(
    (id: string, accepts: string[]) => {
      dropZoneConfig.current.set(id, accepts);
      setDragState((prev) => {
        // Prevent duplicate registrations
        if (prev.dropZones.includes(id)) {
          return prev;
        }
        return {
          ...prev,
          dropZones: [...prev.dropZones, id],
        };
      });
    },
    [],
  );

  const unregisterDropZone = React.useCallback((id: string) => {
    dropZoneConfig.current.delete(id);
    setDragState((prev) => ({
      ...prev,
      dropZones: prev.dropZones.filter((zone) => zone !== id),
    }));
  }, []);

  const value = {
    dragState,
    startDrag,
    updateDrag,
    endDrag,
    registerDropZone,
    unregisterDropZone,
  };

  return (
    <DragDropContext.Provider value={value}>
      {children}
    </DragDropContext.Provider>
  );
};

interface DraggableProps {
  id: string;
  children: React.ReactNode;
  data?: any;
  disabled?: boolean;
}

const Draggable: React.FC<DraggableProps> = ({
  id,
  children,
  data,
  disabled = false,
}) => {
  const context = React.useContext(DragDropContext);
  const elementRef = React.useRef<HTMLDivElement>(null);

  const handleMouseDown = (event: React.MouseEvent) => {
    if (disabled || !context) return;

    const rect = elementRef.current?.getBoundingClientRect();
    if (rect) {
      const offset = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      context.startDrag(id, offset);
    }
  };

  return (
    <div
      ref={elementRef}
      data-testid={`draggable-${id}`}
      className={`draggable ${disabled ? "disabled" : ""} ${context?.dragState.draggedElement === id ? "dragging" : ""}`}
      onMouseDown={handleMouseDown}
      style={{
        cursor: disabled ? "not-allowed" : "grab",
        userSelect: "none",
      }}
    >
      {children}
    </div>
  );
};

interface DroppableProps {
  id: string;
  accepts: string[];
  children: React.ReactNode;
  onDrop?: (itemId: string) => void;
}

const Droppable: React.FC<DroppableProps> = ({
  id,
  accepts,
  children,
  onDrop,
}) => {
  const context = React.useContext(DragDropContext);

  React.useEffect(() => {
    if (context) {
      context.registerDropZone(id, accepts);
      return () => {
        // Use setTimeout to avoid setState during render
        setTimeout(() => {
          context.unregisterDropZone(id);
        }, 0);
      };
    }
  }, [context, id, accepts]);

  const handleMouseUp = () => {
    if (context?.dragState.isDragging) {
      context.endDrag(id);
      onDrop?.(context.dragState.draggedElement!);
    }
  };

  const isValidDropZone =
    context?.dragState.draggedElement &&
    accepts.includes(context.dragState.draggedElement);

  return (
    <div
      data-testid={`droppable-${id}`}
      className={`droppable ${isValidDropZone ? "valid-drop" : ""}`}
      onMouseUp={handleMouseUp}
      style={{
        minHeight: "100px",
        border: `2px dashed ${isValidDropZone ? "#4CAF50" : "#ccc"}`,
        backgroundColor: isValidDropZone ? "#f0f8f0" : "transparent",
      }}
    >
      {children}
    </div>
  );
};

// Tests
describe("Drag and Drop System", () => {
  // Set short timeout for all tests
  jest.setTimeout(1000);

  test("should handle basic drag and drop", () => {
    const onDrop = jest.fn();

    render(
      <DragDropProvider onDrop={onDrop}>
        <Draggable id="item1">
          <div>Draggable Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1"]}>
          <div>Drop Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const droppable = screen.getByTestId("droppable-zone1");

    // Start drag
    fireEvent.mouseDown(draggable, { clientX: 10, clientY: 10 });

    // Drop
    fireEvent.mouseUp(droppable);

    expect(onDrop).toHaveBeenCalledWith("item1", "zone1");
  });

  test("should handle disabled draggable", async () => {
    const onDrop = jest.fn();

    render(
      <DragDropProvider onDrop={onDrop}>
        <Draggable id="item1" disabled>
          <div>Disabled Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1"]}>
          <div>Drop Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const droppable = screen.getByTestId("droppable-zone1");

    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(droppable);

    expect(onDrop).not.toHaveBeenCalled();
  });

  test("should not drop on non-accepting zones", () => {
    const onDrop = jest.fn();

    render(
      <DragDropProvider onDrop={onDrop}>
        <Draggable id="item1">
          <div>Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item2"]}>
          <div>Wrong Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const droppable = screen.getByTestId("droppable-zone1");

    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(droppable);

    expect(onDrop).not.toHaveBeenCalled();
  });

  test("should handle multiple draggable items", () => {
    const onDrop = jest.fn();

    render(
      <DragDropProvider onDrop={onDrop}>
        <Draggable id="item1">
          <div>Item 1</div>
        </Draggable>
        <Draggable id="item2">
          <div>Item 2</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1", "item2"]}>
          <div>Multi Drop Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable1 = screen.getByTestId("draggable-item1");
    const draggable2 = screen.getByTestId("draggable-item2");
    const droppable = screen.getByTestId("droppable-zone1");

    // Test first item
    fireEvent.mouseDown(draggable1);
    fireEvent.mouseUp(droppable);
    expect(onDrop).toHaveBeenCalledWith("item1", "zone1");

    // Test second item
    fireEvent.mouseDown(draggable2);
    fireEvent.mouseUp(droppable);
    expect(onDrop).toHaveBeenCalledWith("item2", "zone1");
  });

  test("should handle multiple drop zones", () => {
    const onDrop = jest.fn();

    render(
      <DragDropProvider onDrop={onDrop}>
        <Draggable id="item1">
          <div>Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1"]}>
          <div>Zone 1</div>
        </Droppable>
        <Droppable id="zone2" accepts={["item1"]}>
          <div>Zone 2</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const droppable1 = screen.getByTestId("droppable-zone1");
    const droppable2 = screen.getByTestId("droppable-zone2");

    // Test drop on first zone
    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(droppable1);
    expect(onDrop).toHaveBeenCalledWith("item1", "zone1");

    // Test drop on second zone
    fireEvent.mouseDown(draggable);
    fireEvent.mouseUp(droppable2);
    expect(onDrop).toHaveBeenCalledWith("item1", "zone2");
  });

  test("should handle drag state changes", () => {
    render(
      <DragDropProvider>
        <Draggable id="item1">
          <div>Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1"]}>
          <div>Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const droppable = screen.getByTestId("droppable-zone1");

    // Start drag - should add dragging class
    fireEvent.mouseDown(draggable);
    expect(draggable).toHaveClass("dragging");

    // End drag - should remove dragging class
    fireEvent.mouseUp(droppable);
    expect(draggable).not.toHaveClass("dragging");
  });

  test("should show valid drop zone styling", () => {
    render(
      <DragDropProvider>
        <Draggable id="item1">
          <div>Item</div>
        </Draggable>
        <Droppable id="zone1" accepts={["item1"]}>
          <div>Valid Zone</div>
        </Droppable>
        <Droppable id="zone2" accepts={["item2"]}>
          <div>Invalid Zone</div>
        </Droppable>
      </DragDropProvider>,
    );

    const draggable = screen.getByTestId("draggable-item1");
    const validZone = screen.getByTestId("droppable-zone1");
    const invalidZone = screen.getByTestId("droppable-zone2");

    // Start drag
    fireEvent.mouseDown(draggable);

    // Valid zone should have valid-drop class
    expect(validZone).toHaveClass("valid-drop");

    // Invalid zone should not have valid-drop class
    expect(invalidZone).not.toHaveClass("valid-drop");
  });
});
