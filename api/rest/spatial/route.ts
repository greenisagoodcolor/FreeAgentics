import { NextRequest, NextResponse } from "next/server";

/**
 * Spatial Grid API Endpoints
 * Handles saving/loading agent positions and movement history
 * Following ADR-008 API design patterns
 */

interface GridPosition {
  agentId: string;
  coordinate: { x: number; y: number };
  proximityRadius: number;
  lastUpdated: string;
  movementHistory?: Array<{
    coordinate: { x: number; y: number };
    timestamp: string;
  }>;
}

interface SpatialGridState {
  gridSize: { width: number; height: number };
  positions: GridPosition[];
  lastSaved: string;
}

// In-memory storage for demo (replace with actual database)
let spatialGridData: Record<string, SpatialGridState> = {};

/**
 * GET /api/rest/spatial
 * Retrieve spatial grid state
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get("sessionId") || "default";

    const gridState = spatialGridData[sessionId];

    if (!gridState) {
      return NextResponse.json({
        success: true,
        data: {
          gridSize: { width: 10, height: 10 },
          positions: [],
          lastSaved: new Date().toISOString(),
        },
      });
    }

    return NextResponse.json({
      success: true,
      data: gridState,
    });
  } catch (error) {
    console.error("Error retrieving spatial grid state:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to retrieve spatial grid state",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * POST /api/rest/spatial
 * Save spatial grid state
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sessionId = "default", gridSize, positions } = body;

    // Validate input
    if (!gridSize || !positions) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required fields: gridSize and positions",
        },
        { status: 400 },
      );
    }

    // Validate grid size
    if (
      !gridSize.width ||
      !gridSize.height ||
      gridSize.width < 1 ||
      gridSize.height < 1 ||
      gridSize.width > 50 ||
      gridSize.height > 50
    ) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid grid size. Must be between 1x1 and 50x50",
        },
        { status: 400 },
      );
    }

    // Validate positions
    for (const position of positions) {
      if (
        !position.agentId ||
        typeof position.coordinate?.x !== "number" ||
        typeof position.coordinate?.y !== "number"
      ) {
        return NextResponse.json(
          {
            success: false,
            error:
              "Invalid position data. Each position must have agentId and coordinate {x, y}",
          },
          { status: 400 },
        );
      }

      // Check bounds
      if (
        position.coordinate.x < 0 ||
        position.coordinate.x >= gridSize.width ||
        position.coordinate.y < 0 ||
        position.coordinate.y >= gridSize.height
      ) {
        return NextResponse.json(
          {
            success: false,
            error: `Position for agent ${position.agentId} is out of bounds`,
          },
          { status: 400 },
        );
      }
    }

    // Save data
    spatialGridData[sessionId] = {
      gridSize,
      positions: positions.map((pos: any) => ({
        ...pos,
        lastUpdated: new Date().toISOString(),
      })),
      lastSaved: new Date().toISOString(),
    };

    return NextResponse.json({
      success: true,
      message: "Spatial grid state saved successfully",
      data: {
        sessionId,
        savedAt: spatialGridData[sessionId].lastSaved,
        positionCount: positions.length,
      },
    });
  } catch (error) {
    console.error("Error saving spatial grid state:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to save spatial grid state",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * PUT /api/rest/spatial
 * Update agent position
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      sessionId = "default",
      agentId,
      coordinate,
      proximityRadius,
    } = body;

    // Validate input
    if (
      !agentId ||
      typeof coordinate?.x !== "number" ||
      typeof coordinate?.y !== "number"
    ) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required fields: agentId and coordinate {x, y}",
        },
        { status: 400 },
      );
    }

    // Get current state
    let gridState = spatialGridData[sessionId];
    if (!gridState) {
      gridState = {
        gridSize: { width: 10, height: 10 },
        positions: [],
        lastSaved: new Date().toISOString(),
      };
      spatialGridData[sessionId] = gridState;
    }

    // Check bounds
    if (
      coordinate.x < 0 ||
      coordinate.x >= gridState.gridSize.width ||
      coordinate.y < 0 ||
      coordinate.y >= gridState.gridSize.height
    ) {
      return NextResponse.json(
        {
          success: false,
          error: "Position is out of grid bounds",
        },
        { status: 400 },
      );
    }

    // Find existing position or create new one
    const existingIndex = gridState.positions.findIndex(
      (pos) => pos.agentId === agentId,
    );
    const timestamp = new Date().toISOString();

    if (existingIndex >= 0) {
      const existingPosition = gridState.positions[existingIndex];

      // Add to movement history
      const movementHistory = existingPosition.movementHistory || [];
      movementHistory.push({
        coordinate: existingPosition.coordinate,
        timestamp: existingPosition.lastUpdated,
      });

      // Keep only last 50 moves
      if (movementHistory.length > 50) {
        movementHistory.splice(0, movementHistory.length - 50);
      }

      // Update position
      gridState.positions[existingIndex] = {
        agentId,
        coordinate,
        proximityRadius:
          proximityRadius || existingPosition.proximityRadius || 2,
        lastUpdated: timestamp,
        movementHistory,
      };
    } else {
      // Add new position
      gridState.positions.push({
        agentId,
        coordinate,
        proximityRadius: proximityRadius || 2,
        lastUpdated: timestamp,
        movementHistory: [],
      });
    }

    gridState.lastSaved = timestamp;

    return NextResponse.json({
      success: true,
      message: "Agent position updated successfully",
      data: {
        agentId,
        coordinate,
        timestamp,
      },
    });
  } catch (error) {
    console.error("Error updating agent position:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to update agent position",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}

/**
 * DELETE /api/rest/spatial
 * Remove agent from spatial grid
 */
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get("sessionId") || "default";
    const agentId = searchParams.get("agentId");

    if (!agentId) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required parameter: agentId",
        },
        { status: 400 },
      );
    }

    const gridState = spatialGridData[sessionId];
    if (!gridState) {
      return NextResponse.json({
        success: true,
        message: "Agent not found (no grid state exists)",
      });
    }

    const initialCount = gridState.positions.length;
    gridState.positions = gridState.positions.filter(
      (pos) => pos.agentId !== agentId,
    );
    const removedCount = initialCount - gridState.positions.length;

    if (removedCount > 0) {
      gridState.lastSaved = new Date().toISOString();
    }

    return NextResponse.json({
      success: true,
      message:
        removedCount > 0 ? "Agent removed successfully" : "Agent not found",
      data: {
        agentId,
        removed: removedCount > 0,
        remainingAgents: gridState.positions.length,
      },
    });
  } catch (error) {
    console.error("Error removing agent from spatial grid:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to remove agent from spatial grid",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
}
