"use client"

import type React from "react"

import { useRef, useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ZoomIn, ZoomOut, RefreshCw, Play, Pause, ChevronDown, ChevronRight } from "lucide-react"
import type { Agent, KnowledgeEntry } from "@/lib/types"
import AboutButton from "./about-button"

interface GlobalKnowledgeGraphProps {
  agents: Agent[]
  onSelectNode: (nodeType: "entry" | "tag", nodeId: string, nodeTitle: string) => void
  onShowAbout: () => void
}

interface Node {
  id: string
  title: string
  x: number
  y: number
  radius: number
  color: string
  type: "entry" | "tag" | "agent"
  agentId?: string
  originalId?: string // For entries, store the original entry ID without the agent prefix
  entryIds?: string[] // For consolidated entries, store all original entry IDs
}

interface Link {
  source: string
  target: string
  strength: number
  color: string
}

interface PhysicsNode extends Node {
  vx: number
  vy: number
  fx: number | null
  fy: number | null
  isPinned?: boolean
}

interface PhysicsLink extends Link {
  source: PhysicsNode
  target: PhysicsNode
}

export default function GlobalKnowledgeGraph({ agents, onSelectNode, onShowAbout }: GlobalKnowledgeGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [nodes, setNodes] = useState<Node[]>([])
  const [links, setLinks] = useState<Link[]>([])
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [selectedNodeInfo, setSelectedNodeInfo] = useState<{
    title: string
    content: string
    type: string
    id: string
  } | null>(null)

  // Store simulation state in refs to avoid re-renders
  const simulationRef = useRef<{
    nodes: PhysicsNode[]
    links: PhysicsLink[]
  } | null>(null)

  const [isSimulationRunning, setIsSimulationRunning] = useState(false)
  const [physicsSettings, setPhysicsSettings] = useState({
    repulsion: 80,
    linkStrength: 0.08,
    friction: 0.92,
    centerForce: 0.05,
    collisionRadius: 1.2,
    velocityLimit: 0.2,
  })
  const [lastClickTime, setLastClickTime] = useState(0)
  const [lastClickedNode, setLastClickedNode] = useState<string | null>(null)
  const [draggedNode, setDraggedNode] = useState<PhysicsNode | null>(null)
  const [showSettings, setShowSettings] = useState(false)

  // Store these values in refs to avoid re-renders
  const coolingRef = useRef(1.0)
  const warmupPhaseRef = useRef(0.3)
  const lowMovementFramesRef = useRef(0)
  const animationFrameRef = useRef<number>()
  const initialNodesRef = useRef<Node[]>([])
  const initialLinksRef = useRef<Link[]>([])
  const needsRenderRef = useRef(false)
  const renderIntervalRef = useRef<NodeJS.Timeout>()
  const hasInitializedRef = useRef(false)

  // Initialize the graph data
  useEffect(() => {
    if (!agents.length) return

    // Get container dimensions for better initial positioning
    const container = containerRef.current
    const width = container?.clientWidth || 800
    const height = container?.clientHeight || 600
    const centerX = width / 2
    const centerY = height / 2

    // Extract all knowledge entries and tags
    const allEntries: Array<{ entry: KnowledgeEntry; agentId: string; agentColor: string }> = []
    const allTags = new Set<string>()

    agents.forEach((agent) => {
      agent.knowledge.forEach((entry) => {
        allEntries.push({
          entry,
          agentId: agent.id,
          agentColor: agent.color,
        })
        entry.tags.forEach((tag) => allTags.add(tag))
      })
    })

    // Consolidate knowledge entries by title
    const uniqueEntryTitles = new Map<
      string,
      {
        entryIds: string[]
        agentIds: string[]
        color: string
      }
    >()

    allEntries.forEach(({ entry, agentId, agentColor }) => {
      if (!uniqueEntryTitles.has(entry.title)) {
        uniqueEntryTitles.set(entry.title, {
          entryIds: [entry.id],
          agentIds: [agentId],
          color: "#a855f7", // Purple for consolidated entries
        })
      } else {
        const current = uniqueEntryTitles.get(entry.title)!
        current.entryIds.push(entry.id)
        if (!current.agentIds.includes(agentId)) {
          current.agentIds.push(agentId)
        }
      }
    })

    // Create nodes for agents, consolidated entries and tags
    const newNodes: Node[] = [
      // Agent nodes in the center
      ...agents.map((agent, index) => {
        const angle = (index / agents.length) * Math.PI * 2
        const radius = Math.min(width, height) * 0.15 // 15% of the smaller dimension
        return {
          id: `agent-${agent.id}`,
          title: agent.name,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          radius: 18,
          color: agent.color,
          type: "agent" as const,
          agentId: agent.id,
        }
      }),

      // Consolidated entry nodes in a middle circle
      ...[...uniqueEntryTitles.entries()].map(([title, data], index) => {
        const angle = (index / uniqueEntryTitles.size) * Math.PI * 2
        const radius = Math.min(width, height) * 0.3 // 30% of the smaller dimension
        return {
          id: `entry-${title.replace(/\s+/g, "-").toLowerCase()}`,
          title: title,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          radius: 12,
          color: data.color,
          type: "entry" as const,
          entryIds: data.entryIds,
        }
      }),

      // Tag nodes in an outer circle
      ...[...allTags].map((tag, index) => {
        const angle = (index / allTags.size) * Math.PI * 2
        const radius = Math.min(width, height) * 0.45 // 45% of the smaller dimension
        return {
          id: `tag-${tag}`,
          title: tag,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          radius: 10, // Increased from 8 to make tags more visible
          color: "#6366f1", // Indigo for tags
          type: "tag" as const,
        }
      }),
    ]

    // Create links
    const newLinks: Link[] = []

    // Links between agents and their entries (now consolidated)
    agents.forEach((agent) => {
      agent.knowledge.forEach((entry) => {
        const entryNodeId = `entry-${entry.title.replace(/\s+/g, "-").toLowerCase()}`

        // Link agent to entry
        newLinks.push({
          source: `agent-${agent.id}`,
          target: entryNodeId,
          strength: 0.7,
          color: agent.color,
        })

        // Links between entries and their tags
        entry.tags.forEach((tag) => {
          // Check if this link already exists to avoid duplicates
          const linkExists = newLinks.some(
            (link) =>
              (link.source === entryNodeId && link.target === `tag-${tag}`) ||
              (link.source === `tag-${tag}` && link.target === entryNodeId),
          )

          if (!linkExists) {
            newLinks.push({
              source: entryNodeId,
              target: `tag-${tag}`,
              strength: 0.5,
              color: "#a855f7", // Purple for all tag connections
            })
          }
        })
      })
    })

    // Links between entries that share tags
    const entryNodes = newNodes.filter((node) => node.type === "entry")
    for (let i = 0; i < entryNodes.length; i++) {
      for (let j = i + 1; j < entryNodes.length; j++) {
        const entry1 = entryNodes[i]
        const entry2 = entryNodes[j]

        // Find all entries with these titles to get their tags
        const entry1Tags = new Set<string>()
        const entry2Tags = new Set<string>()

        allEntries.forEach(({ entry }) => {
          if (entry.title === entry1.title) {
            entry.tags.forEach((tag) => entry1Tags.add(tag))
          }
          if (entry.title === entry2.title) {
            entry.tags.forEach((tag) => entry2Tags.add(tag))
          }
        })

        // Find shared tags
        const sharedTags = [...entry1Tags].filter((tag) => entry2Tags.has(tag))

        if (sharedTags.length > 0) {
          newLinks.push({
            source: entry1.id,
            target: entry2.id,
            strength: 0.3 * sharedTags.length,
            color: "#a855f7", // Purple for shared knowledge
          })
        }
      }
    }

    // Store the initial nodes and links for reset functionality
    // Deep clone to ensure we have completely separate objects
    initialNodesRef.current = JSON.parse(JSON.stringify(newNodes))
    initialLinksRef.current = JSON.parse(JSON.stringify(newLinks))
    hasInitializedRef.current = true

    setNodes(newNodes)
    setLinks(newLinks)
  }, [agents])

  // Initialize physics simulation
  useEffect(() => {
    if (!nodes.length || !links.length) return

    // Create physics nodes with velocity properties
    const physicsNodes = nodes.map((node) => ({
      ...node,
      vx: 0, // Velocity X
      vy: 0, // Velocity Y
      fx: null, // Fixed X (for pinned nodes)
      fy: null, // Fixed Y (for pinned nodes)
      isPinned: false, // Whether the node is pinned in place
    }))

    // Create physics links with actual node references instead of just IDs
    const physicsLinks = links
      .map((link) => {
        const source = physicsNodes.find((n) => n.id === link.source)
        const target = physicsNodes.find((n) => n.id === link.target)

        if (!source || !target) {
          console.error(`Could not find nodes for link: ${link.source} -> ${link.target}`)
          return null
        }

        return {
          ...link,
          source,
          target,
        }
      })
      .filter(Boolean) as PhysicsLink[]

    // Store in ref instead of state to avoid re-renders
    simulationRef.current = {
      nodes: physicsNodes,
      links: physicsLinks,
    }
  }, [nodes, links])

  // Simple quadtree implementation for spatial partitioning
  class QuadTree {
    boundary: { x: number; y: number; width: number; height: number }
    capacity: number
    points: Array<{ x: number; y: number; node: any }>
    divided: boolean
    northeast: QuadTree | null
    northwest: QuadTree | null
    southeast: QuadTree | null
    southwest: QuadTree | null

    constructor(boundary: { x: number; y: number; width: number; height: number }, capacity: number) {
      this.boundary = boundary
      this.capacity = capacity
      this.points = []
      this.divided = false
      this.northeast = null
      this.northwest = null
      this.southeast = null
      this.southwest = null
    }

    insert(point: { x: number; y: number; node: any }): boolean {
      // Check if point is in boundary
      if (!this.contains(point)) {
        return false
      }

      // If space available, add point
      if (this.points.length < this.capacity) {
        this.points.push(point)
        return true
      }

      // Otherwise, subdivide and add point to appropriate quadrant
      if (!this.divided) {
        this.subdivide()
      }

      if (this.northeast!.insert(point)) return true
      if (this.northwest!.insert(point)) return true
      if (this.southeast!.insert(point)) return true
      if (this.southwest!.insert(point)) return false

      return false
    }

    subdivide() {
      const x = this.boundary.x
      const y = this.boundary.y
      const w = this.boundary.width / 2
      const h = this.boundary.height / 2

      this.northeast = new QuadTree({ x: x + w, y: y - h, width: w, height: h }, this.capacity)
      this.northwest = new QuadTree({ x: x - w, y: y - h, width: w, height: h }, this.capacity)
      this.southeast = new QuadTree({ x: x + w, y: y + h, width: w, height: h }, this.capacity)
      this.southwest = new QuadTree({ x: x - w, y: y + h, width: w, height: h }, this.capacity)

      this.divided = true
    }

    contains(point: { x: number; y: number }): boolean {
      return (
        point.x >= this.boundary.x - this.boundary.width &&
        point.x <= this.boundary.x + this.boundary.width &&
        point.y >= this.boundary.y - this.boundary.height &&
        point.y <= this.boundary.y + this.boundary.height
      )
    }

    query(range: { x: number; y: number; radius: number }, found: Array<any> = []): Array<any> {
      // Check if range intersects boundary
      if (!this.intersects(range)) {
        return found
      }

      // Check points in this quad
      for (const point of this.points) {
        const dx = range.x - point.x
        const dy = range.y - point.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance <= range.radius) {
          found.push(point.node)
        }
      }

      // If this quad is divided, check children
      if (this.divided) {
        this.northeast!.query(range, found)
        this.northwest!.query(range, found)
        this.southeast!.query(range, found)
        this.southwest!.query(range, found)
      }

      return found
    }

    intersects(range: { x: number; y: number; radius: number }): boolean {
      const dx = Math.abs(range.x - this.boundary.x)
      const dy = Math.abs(range.y - this.boundary.y)

      if (dx > this.boundary.width + range.radius) return false
      if (dy > this.boundary.height + range.radius) return false

      if (dx <= this.boundary.width) return true
      if (dy <= this.boundary.height) return true

      const cornerDistanceSq =
        (dx - this.boundary.width) * (dx - this.boundary.width) +
        (dy - this.boundary.height) * (dy - this.boundary.height)

      return cornerDistanceSq <= range.radius * range.radius
    }
  }

  // Apply repulsion forces between all nodes
  const applyRepulsionForces = useCallback(() => {
    const simulation = simulationRef.current
    if (!simulation) return

    const nodes = simulation.nodes
    const container = containerRef.current
    if (!container) return

    // Create quadtree
    const boundary = {
      x: container.clientWidth / 2,
      y: container.clientHeight / 2,
      width: container.clientWidth,
      height: container.clientHeight,
    }

    const quadtree = new QuadTree(boundary, 4)

    // Insert all nodes into quadtree
    nodes.forEach((node) => {
      quadtree.insert({ x: node.x, y: node.y, node })
    })

    // Calculate repulsion using quadtree for optimization
    nodes.forEach((nodeA) => {
      // Find nodes within a certain radius
      const radius = Math.max(100, nodeA.radius * 10) // Adjust radius as needed
      const nearbyNodes = quadtree.query({ x: nodeA.x, y: nodeA.y, radius })

      nearbyNodes.forEach((nodeB) => {
        if (nodeA === nodeB) return

        // Calculate distance
        const dx = nodeB.x - nodeA.x
        const dy = nodeB.y - nodeA.y
        const distanceSq = dx * dx + dy * dy
        const distance = Math.sqrt(distanceSq)

        if (distance === 0) return

        // Calculate repulsion force with a minimum distance to prevent extreme forces
        const minDistance = nodeA.radius + nodeB.radius
        const effectiveDistance = Math.max(distance, minDistance)

        // Use a softer inverse law (1/d instead of 1/d²) for more stability
        const force = (physicsSettings.repulsion * warmupPhaseRef.current) / effectiveDistance

        // Apply force to velocity with dampening for stability
        const forceX = (dx / distance) * force * 0.5
        const forceY = (dy / distance) * force * 0.5

        nodeA.vx -= forceX
        nodeA.vy -= forceY
      })
    })
  }, [physicsSettings.repulsion])

  // Apply attraction forces between linked nodes
  const applyAttractionForces = useCallback(() => {
    const simulation = simulationRef.current
    if (!simulation) return

    simulation.links.forEach((link) => {
      const source = link.source
      const target = link.target

      // Calculate distance
      const dx = target.x - source.x
      const dy = target.y - source.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      if (distance === 0) return

      // Calculate attraction force
      const force = distance * physicsSettings.linkStrength * link.strength * warmupPhaseRef.current

      // Apply force to velocity
      const forceX = (dx / distance) * force
      const forceY = (dy / distance) * force

      source.vx += forceX
      source.vy += forceY
      target.vx -= forceX
      target.vy -= forceY
    })
  }, [physicsSettings.linkStrength])

  // Apply a force to keep nodes near the center
  const applyCenteringForce = useCallback(() => {
    const simulation = simulationRef.current
    if (!simulation) return

    const container = containerRef.current
    if (!container) return

    const centerX = container.clientWidth / 2
    const centerY = container.clientHeight / 2

    simulation.nodes.forEach((node) => {
      // Calculate distance from center
      const dx = centerX - node.x
      const dy = centerY - node.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      if (distance === 0) return

      // Apply centering force (stronger for nodes far from center)
      const force = distance * physicsSettings.centerForce * warmupPhaseRef.current

      // Apply force to velocity
      const forceX = (dx / distance) * force
      const forceY = (dy / distance) * force

      node.vx += forceX
      node.vy += forceY
    })
  }, [physicsSettings.centerForce])

  // Prevent nodes from overlapping
  const applyCollisionAvoidance = useCallback(() => {
    const simulation = simulationRef.current
    if (!simulation) return

    const nodes = simulation.nodes

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const nodeA = nodes[i]
        const nodeB = nodes[j]

        // Calculate distance
        const dx = nodeB.x - nodeA.x
        const dy = nodeB.y - nodeA.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        // Calculate minimum distance to avoid collision
        const minDistance = (nodeA.radius + nodeB.radius) * physicsSettings.collisionRadius

        if (distance < minDistance && distance > 0) {
          // Calculate overlap
          const overlap = (minDistance - distance) / distance

          // Apply force to separate nodes
          const moveX = dx * overlap * 0.5
          const moveY = dy * overlap * 0.5

          // Only move nodes that aren't pinned
          if (!nodeA.isPinned) {
            nodeA.x -= moveX
            nodeA.y -= moveY
          }

          if (!nodeB.isPinned) {
            nodeB.x += moveX
            nodeB.y += moveY
          }
        }
      }
    }
  }, [physicsSettings.collisionRadius])

  const runSimulation = useCallback(() => {
    const simulation = simulationRef.current
    if (!simulation || !isSimulationRunning) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      return
    }

    // Apply forces
    applyRepulsionForces()
    applyAttractionForces()
    applyCenteringForce()

    // Update positions
    let totalMovement = 0

    simulation.nodes.forEach((node) => {
      // Skip pinned nodes
      if (node.isPinned || node.fx !== null || node.fy !== null) {
        node.vx = 0
        node.vy = 0
        return
      }

      // Apply velocity with friction and cooling
      node.vx *= physicsSettings.friction * coolingRef.current
      node.vy *= physicsSettings.friction * coolingRef.current

      // Limit velocity to prevent extreme movements
      const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy)
      if (speed > physicsSettings.velocityLimit) {
        node.vx = (node.vx / speed) * physicsSettings.velocityLimit
        node.vy = (node.vy / speed) * physicsSettings.velocityLimit
      }

      // Update position
      node.x += node.vx
      node.y += node.vy

      // Track total movement for cooling
      totalMovement += Math.abs(node.vx) + Math.abs(node.vy)
    })

    // Apply collision avoidance after position updates
    applyCollisionAvoidance()

    // Mark that we need to render
    needsRenderRef.current = true

    // Auto-stop simulation if movement is very small for a sustained period
    if (totalMovement < 0.05) {
      // Count low movement frames instead of stopping immediately
      lowMovementFramesRef.current++
      if (lowMovementFramesRef.current > 30) {
        // About 0.5 seconds of low movement
        setIsSimulationRunning(false)
        lowMovementFramesRef.current = 0
        return
      }
    } else {
      lowMovementFramesRef.current = 0
    }

    // Continue animation loop
    animationFrameRef.current = requestAnimationFrame(runSimulation)
  }, [
    isSimulationRunning,
    applyRepulsionForces,
    applyAttractionForces,
    applyCenteringForce,
    applyCollisionAvoidance,
    physicsSettings.friction,
    physicsSettings.velocityLimit,
  ])

  // Set up a separate interval for updating the React state
  useEffect(() => {
    if (isSimulationRunning) {
      // Start the simulation
      runSimulation()

      // Set up an interval to update the React state less frequently
      renderIntervalRef.current = setInterval(() => {
        if (needsRenderRef.current && simulationRef.current) {
          // Create a copy of the nodes to avoid mutating the original
          const updatedNodes = simulationRef.current.nodes.map((node) => ({
            ...node,
          }))

          // Update the React state
          setNodes(updatedNodes)

          // Reset the flag
          needsRenderRef.current = false
        }
      }, 50) // Update every 50ms (20fps) instead of every frame

      // Cooling effect
      coolingRef.current = 1.0
      const coolingInterval = setInterval(() => {
        coolingRef.current = Math.max(coolingRef.current * 0.98, 0.6)
      }, 500)

      // Warmup effect
      warmupPhaseRef.current = 0.3
      const warmupInterval = setInterval(() => {
        warmupPhaseRef.current = Math.min(warmupPhaseRef.current + 0.1, 1.0)
      }, 100)

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current)
        }
        if (renderIntervalRef.current) {
          clearInterval(renderIntervalRef.current)
        }
        clearInterval(coolingInterval)
        clearInterval(warmupInterval)
      }
    } else {
      // Clean up when simulation is stopped
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (renderIntervalRef.current) {
        clearInterval(renderIntervalRef.current)
      }
    }
  }, [isSimulationRunning, runSimulation])

  const resetNodePositions = useCallback(() => {
    if (!hasInitializedRef.current || initialNodesRef.current.length === 0) {
      console.warn("Cannot reset: initial nodes not stored")
      return
    }

    // Stop simulation
    setIsSimulationRunning(false)
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    if (renderIntervalRef.current) {
      clearInterval(renderIntervalRef.current)
    }

    // Reset zoom and offset
    setZoomLevel(1)
    setOffset({ x: 0, y: 0 })

    // Clear any selected or hovered nodes
    setSelectedNode(null)
    setHoveredNode(null)

    // Deep clone the initial nodes and links to ensure we're working with fresh copies
    const resetNodes = JSON.parse(JSON.stringify(initialNodesRef.current))
    const resetLinks = JSON.parse(JSON.stringify(initialLinksRef.current))

    // Update the state with the initial nodes and links
    setNodes(resetNodes)
    setLinks(resetLinks)

    // Reset the simulation with fresh physics nodes
    const physicsNodes = resetNodes.map((node) => ({
      ...node,
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
      isPinned: false,
    }))

    // Create physics links with actual node references
    const physicsLinks = resetLinks
      .map((link) => {
        const source = physicsNodes.find((n) => n.id === link.source)
        const target = physicsNodes.find((n) => n.id === link.target)

        if (!source || !target) {
          console.error(`Could not find nodes for link: ${link.source} -> ${link.target}`)
          return null
        }

        return {
          ...link,
          source,
          target,
        }
      })
      .filter(Boolean) as PhysicsLink[]

    // Update the simulation reference
    simulationRef.current = {
      nodes: physicsNodes,
      links: physicsLinks,
    }

    // Reset other simulation parameters
    coolingRef.current = 1.0
    warmupPhaseRef.current = 0.3
    lowMovementFramesRef.current = 0

    // Force a render
    needsRenderRef.current = true
  }, [])

  const handleNodeClick = useCallback(
    (node: Node) => {
      if (!node) return

      if (node.type === "tag") {
        const tagName = node.title

        // Find all knowledge entries that have this tag
        const entriesWithTag: { entry: KnowledgeEntry; agent: Agent }[] = []

        agents.forEach((agent) => {
          agent.knowledge.forEach((entry) => {
            if (entry.tags.includes(tagName)) {
              entriesWithTag.push({ entry, agent })
            }
          })
        })

        if (entriesWithTag.length > 0) {
          // Create a formatted list of entries with this tag
          const formattedEntries = entriesWithTag
            .map(({ entry, agent }) => `- "${entry.title}" (${agent.name})`)
            .join("\n")

          setSelectedNodeInfo({
            title: `Tag: ${tagName}`,
            content: `${entriesWithTag.length} knowledge ${
              entriesWithTag.length === 1 ? "entry has" : "entries have"
            } this tag:\n\n${formattedEntries}`,
            type: "tag",
            id: tagName,
          })
        } else {
          setSelectedNodeInfo({
            title: `Tag: ${tagName}`,
            content: "No knowledge entries have this tag.",
            type: "tag",
            id: tagName,
          })
        }

        onSelectNode("tag", tagName, tagName)
      } else if (node.type === "entry") {
        // For entries, we need to find the actual entry data
        const entryTitle = node.title
        const entriesWithTitle: { entry: KnowledgeEntry; agent: Agent }[] = []

        agents.forEach((agent) => {
          agent.knowledge.forEach((entry) => {
            if (entry.title === entryTitle) {
              entriesWithTitle.push({ entry, agent })
            }
          })
        })

        if (entriesWithTitle.length > 0) {
          // Just use the first one for now for the info panel
          const { entry, agent } = entriesWithTitle[0]

          setSelectedNodeInfo({
            title: entry.title,
            content: entry.content,
            type: "entry",
            id: entry.id,
          })

          // Pass the entry title as the ID to ensure we can find all instances across agents
          onSelectNode("entry", entryTitle, entry.title)
        }
      }
    },
    [agents, onSelectNode],
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const x = (e.clientX - rect.left - offset.x) / zoomLevel
      const y = (e.clientY - rect.top - offset.y) / zoomLevel

      // Check if clicking on a node
      const clicked = nodes.find((node) => {
        const dx = node.x - x
        const dy = node.y - y
        return Math.sqrt(dx * dx + dy * dy) <= node.radius
      })

      if (clicked && simulationRef.current) {
        setSelectedNode(clicked)
        handleNodeClick(clicked)

        // Check for double-click to pin/unpin node
        const now = Date.now()
        if (now - lastClickTime < 300 && lastClickedNode === clicked.id) {
          // Toggle pin state
          const simNode = simulationRef.current.nodes.find((n) => n.id === clicked.id)
          if (simNode) {
            simNode.isPinned = !simNode.isPinned
            simNode.fx = simNode.isPinned ? simNode.x : null
            simNode.fy = simNode.isPinned ? simNode.y : null

            // Update the React state
            needsRenderRef.current = true
          }
        } else {
          // Start dragging the node
          const simNode = simulationRef.current.nodes.find((n) => n.id === clicked.id)
          if (simNode) {
            setDraggedNode(simNode)
          }
        }

        setLastClickTime(now)
        setLastClickedNode(clicked.id)
      } else {
        // Start dragging the canvas
        setIsDragging(true)
        setDragStart({
          x: e.clientX - offset.x,
          y: e.clientY - offset.y,
        })
      }
    },
    [nodes, zoomLevel, offset, lastClickTime, lastClickedNode, handleNodeClick],
  )

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const x = (e.clientX - rect.left - offset.x) / zoomLevel
      const y = (e.clientY - rect.top - offset.y) / zoomLevel

      // Check if hovering over a node
      const hovered = nodes.find((node) => {
        const dx = node.x - x
        const dy = node.y - y
        return Math.sqrt(dx * dx + dy * dy) <= node.radius
      })

      setHoveredNode(hovered || null)

      // Handle dragging a node
      if (draggedNode && simulationRef.current) {
        const nodeIndex = simulationRef.current.nodes.findIndex((n) => n.id === draggedNode.id)
        if (nodeIndex >= 0) {
          const node = simulationRef.current.nodes[nodeIndex]
          node.x = x
          node.y = y
          node.vx = 0
          node.vy = 0

          // Mark that we need to render
          needsRenderRef.current = true
        }
      } else if (isDragging) {
        // Handle dragging the canvas
        setOffset({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y,
        })
      }
    },
    [nodes, zoomLevel, offset, dragStart, draggedNode, isDragging],
  )

  const handleCanvasMouseUp = useCallback(() => {
    setIsDragging(false)
    setDraggedNode(null)
  }, [])

  const handleZoomIn = useCallback(() => {
    setZoomLevel((prev) => Math.min(prev + 0.2, 3))
  }, [])

  const handleZoomOut = useCallback(() => {
    setZoomLevel((prev) => Math.max(prev - 0.2, 0.5))
  }, [])

  const handleReset = useCallback(() => {
    setZoomLevel(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  // Draw the graph
  useEffect(() => {
    if (!nodes.length) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const container = containerRef.current
    if (container) {
      const { width, height } = container.getBoundingClientRect()
      canvas.width = width
      canvas.height = height
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Apply zoom and pan
    ctx.save()
    ctx.translate(offset.x, offset.y)
    ctx.scale(zoomLevel, zoomLevel)

    // Draw links
    for (const link of links) {
      const source = nodes.find((n) => n.id === link.source)
      const target = nodes.find((n) => n.id === link.target)

      if (source && target) {
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)

        // Use different style for links connected to selected node
        if (selectedNode && (source.id === selectedNode.id || target.id === selectedNode.id)) {
          ctx.strokeStyle = `${link.color}90` // 90% opacity
          ctx.lineWidth = 2 / zoomLevel
        } else {
          ctx.strokeStyle = `${link.color}50` // 50% opacity
          ctx.lineWidth = 1 / zoomLevel
        }

        ctx.stroke()
      }
    }

    // Draw nodes
    for (const node of nodes) {
      // Ensure radius is positive
      const radius = Math.max(node.radius, 1) // Minimum radius of 1

      ctx.beginPath()
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)

      // Highlight selected or hovered node
      if (node === selectedNode) {
        ctx.fillStyle = "#f472b6" // Pink for selected
        ctx.strokeStyle = "#ffffff"
        ctx.lineWidth = 2 / zoomLevel
        ctx.stroke()
      } else if (node === hoveredNode) {
        ctx.fillStyle = node.color
        ctx.strokeStyle = "#ffffff"
        ctx.lineWidth = 2 / zoomLevel
        ctx.stroke()
      } else {
        ctx.fillStyle = node.color
      }

      // Add a stroke to tag nodes to make them more distinguishable
      if (node.type === "tag") {
        ctx.strokeStyle = "#4338ca"
        ctx.lineWidth = 2 / zoomLevel
        ctx.stroke()
      }

      // Add a pin indicator for pinned nodes
      const isPinned = simulationRef.current?.nodes.find((n) => n.id === node.id)?.isPinned
      if (isPinned) {
        ctx.strokeStyle = "#f59e0b" // Amber color for pins
        ctx.lineWidth = 2 / zoomLevel
        ctx.setLineDash([3, 3]) // Dashed line
        ctx.stroke()
        ctx.setLineDash([]) // Reset to solid line
      }

      ctx.fill()

      // Draw agent initials for agent nodes
      if (node.type === "agent") {
        ctx.fillStyle = "#ffffff"
        ctx.font = `bold ${Math.max(14 / zoomLevel, 8)}px Arial` // Ensure minimum font size
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(node.title.charAt(0), node.x, node.y)
      }

      // Draw node labels
      if (node === hoveredNode || node === selectedNode || zoomLevel > 1.5) {
        ctx.fillStyle = "#ffffff"
        ctx.font = `${Math.max(12 / zoomLevel, 8)}px Arial` // Ensure minimum font size
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"

        // Draw text with background for better readability
        const textWidth = ctx.measureText(node.title).width
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
        ctx.fillRect(node.x - textWidth / 2 - 4, node.y + radius + 4, textWidth + 8, 16 / zoomLevel)

        ctx.fillStyle = "#ffffff"
        ctx.font = `${Math.max(12 / zoomLevel, 8)}px Arial`
        ctx.fontWeight = node.type === "tag" ? "bold" : "normal"
        ctx.fillText(node.title, node.x, node.y + radius + 12 / zoomLevel)
      }
    }

    ctx.restore()
  }, [nodes, links, hoveredNode, selectedNode, zoomLevel, offset])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return

      canvas.width = container.clientWidth
      canvas.height = container.clientHeight

      // Redraw
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
    }

    window.addEventListener("resize", handleResize)
    return () => {
      window.removeEventListener("resize", handleResize)
    }
  }, [])

  // Count total knowledge entries
  const totalEntries = agents.reduce((sum, agent) => sum + agent.knowledge.length, 0)

  // Count unique knowledge titles
  const uniqueTitles = new Set<string>()
  agents.forEach((agent) => {
    agent.knowledge.forEach((entry) => {
      uniqueTitles.add(entry.title)
    })
  })

  // Count unique tags
  const uniqueTags = new Set<string>()
  agents.forEach((agent) => {
    agent.knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => uniqueTags.add(tag))
    })
  })

  return (
    <Card className="h-full">
      <CardHeader className="py-2 px-4 border-b border-purple-800 bg-gradient-to-r from-purple-900/50 to-indigo-900/50">
        <div className="flex justify-between items-center">
          <CardTitle className="text-sm font-medium text-white">Global Knowledge Graph</CardTitle>
          <div className="flex gap-2">
            {/* Physics simulation controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsSimulationRunning(!isSimulationRunning)}
              className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
              title={isSimulationRunning ? "Pause simulation" : "Start simulation"}
            >
              {isSimulationRunning ? <Pause size={12} /> : <Play size={12} />}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={resetNodePositions}
              className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
              title="Reset positions"
            >
              <RefreshCw size={12} />
            </Button>

            {/* Existing zoom controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleZoomOut}
              className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <ZoomOut size={12} />
            </Button>
            <span className="flex items-center text-xs text-white px-1">{Math.round(zoomLevel * 100)}%</span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleZoomIn}
              className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <ZoomIn size={12} />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              className="h-6 w-6 p-0 bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <RefreshCw size={12} />
            </Button>
          </div>
        </div>
        <div className="text-xs text-purple-300 mt-1">
          {agents.length} agents, {totalEntries} entries ({uniqueTitles.size} unique), {uniqueTags.size} tags
          {isSimulationRunning && " • Simulation running"}
          {simulationRef.current?.nodes.filter((n) => n.isPinned).length > 0 &&
            ` • ${simulationRef.current.nodes.filter((n) => n.isPinned).length} pinned nodes`}
        </div>
      </CardHeader>

      <CardContent className="p-0 h-[calc(100%-52px)]">
        <div ref={containerRef} className="h-full relative bg-black">
          <canvas
            ref={canvasRef}
            className="absolute inset-0 cursor-grab"
            onMouseMove={handleCanvasMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseUp}
          />

          {hoveredNode && (
            <div className="absolute bottom-4 left-4 bg-purple-950/80 backdrop-blur-sm p-3 rounded-lg border border-purple-700 shadow-md max-w-xs">
              {hoveredNode.type === "tag" && (
                <>
                  <h3 className="font-medium text-white mb-1">Tag: {hoveredNode.title}</h3>
                  <div className="text-xs text-purple-300">
                    {agents.reduce((count, agent) => {
                      return count + agent.knowledge.filter((entry) => entry.tags.includes(hoveredNode.title)).length
                    }, 0)}{" "}
                    entries across {agents.length} agents
                  </div>
                </>
              )}

              {hoveredNode.type === "entry" && (
                <>
                  <h3 className="font-medium text-white mb-1">Knowledge: {hoveredNode.title}</h3>
                  <div className="text-xs text-purple-300">
                    {agents.reduce((count, agent) => {
                      return count + agent.knowledge.filter((entry) => entry.title === hoveredNode.title).length
                    }, 0)}{" "}
                    instances across{" "}
                    {
                      agents.filter((agent) => agent.knowledge.some((entry) => entry.title === hoveredNode.title))
                        .length
                    }{" "}
                    agents
                  </div>
                </>
              )}
            </div>
          )}
        </div>
        <div className="absolute bottom-4 right-4 bg-purple-950/80 backdrop-blur-sm rounded-lg border border-purple-700 shadow-md overflow-hidden z-50">
          <div className="flex items-center">
            <AboutButton onClick={onShowAbout} />
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center justify-between p-2 text-white text-sm font-medium hover:bg-purple-800/50"
            >
              Physics Settings
              {showSettings ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
          </div>

          {showSettings && (
            <div className="p-3 space-y-3">
              <div>
                <label className="text-xs text-purple-300 block mb-1">Repulsion Force</label>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={physicsSettings.repulsion}
                  onChange={(e) =>
                    setPhysicsSettings({
                      ...physicsSettings,
                      repulsion: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-purple-400">
                  <span>Weak</span>
                  <span>{physicsSettings.repulsion}</span>
                  <span>Strong</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-purple-300 block mb-1">Link Strength</label>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={physicsSettings.linkStrength}
                  onChange={(e) =>
                    setPhysicsSettings({
                      ...physicsSettings,
                      linkStrength: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-purple-400">
                  <span>Loose</span>
                  <span>{physicsSettings.linkStrength.toFixed(2)}</span>
                  <span>Tight</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-purple-300 block mb-1">Friction</label>
                <input
                  type="range"
                  min="0.7"
                  max="0.99"
                  step="0.01"
                  value={physicsSettings.friction}
                  onChange={(e) =>
                    setPhysicsSettings({
                      ...physicsSettings,
                      friction: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-purple-400">
                  <span>More Damping</span>
                  <span>{physicsSettings.friction.toFixed(2)}</span>
                  <span>Less Damping</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-purple-300 block mb-1">Center Force</label>
                <input
                  type="range"
                  min="0"
                  max="0.3"
                  step="0.01"
                  value={physicsSettings.centerForce}
                  onChange={(e) =>
                    setPhysicsSettings({
                      ...physicsSettings,
                      centerForce: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-purple-400">
                  <span>None</span>
                  <span>{physicsSettings.centerForce.toFixed(2)}</span>
                  <span>Strong</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-purple-300 block mb-1">Velocity Limit</label>
                <input
                  type="range"
                  min="0.0"
                  max="2.5"
                  step="0.1"
                  value={physicsSettings.velocityLimit}
                  onChange={(e) =>
                    setPhysicsSettings({
                      ...physicsSettings,
                      velocityLimit: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-purple-400">
                  <span>Slow</span>
                  <span>{physicsSettings.velocityLimit.toFixed(1)}</span>
                  <span>Fast</span>
                </div>
              </div>

              <button
                onClick={() => {
                  // Reset to default settings
                  setPhysicsSettings({
                    repulsion: 80,
                    linkStrength: 0.08,
                    friction: 0.92,
                    centerForce: 0.05,
                    collisionRadius: 1.2,
                    velocityLimit: 0.2,
                  })
                }}
                className="w-full py-1 px-2 bg-purple-700 hover:bg-purple-600 text-white text-xs rounded"
              >
                Reset to Defaults
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
