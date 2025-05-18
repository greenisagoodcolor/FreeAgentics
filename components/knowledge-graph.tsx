"use client"

import type React from "react"

import { useRef, useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ZoomIn, ZoomOut } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { KnowledgeEntry } from "@/lib/types"

interface KnowledgeGraphProps {
  knowledge: KnowledgeEntry[]
  onSelectEntry: (entry: KnowledgeEntry) => void
  selectedEntry: KnowledgeEntry | null
}

interface Node {
  id: string
  title: string
  x: number
  y: number
  radius: number
  color: string
  type: "entry" | "tag"
}

interface Link {
  source: string
  target: string
  strength: number
}

export default function KnowledgeGraph({ knowledge, onSelectEntry, selectedEntry }: KnowledgeGraphProps) {
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
  const [debugInfo, setDebugInfo] = useState({ width: 0, height: 0, nodeCount: 0 })
  const [showDebug, setShowDebug] = useState(false)

  // Initialize the graph data
  useEffect(() => {
    if (!knowledge.length) return

    // Get container dimensions for better initial positioning
    const container = containerRef.current
    const width = container?.clientWidth || 800
    const height = container?.clientHeight || 600
    const centerX = width / 2
    const centerY = height / 2

    // Extract all unique tags
    const allTags = new Set<string>()
    knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => allTags.add(tag))
    })

    // Create nodes for entries and tags with fixed initial positions
    const newNodes: Node[] = [
      // Entry nodes in an inner circle
      ...knowledge.map((entry, index) => {
        const angle = (index / knowledge.length) * Math.PI * 2
        const radius = Math.min(width, height) * 0.25 // 25% of the smaller dimension
        return {
          id: entry.id,
          title: entry.title,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          radius: 15,
          color: "#a855f7", // Purple for entries
          type: "entry" as const,
        }
      }),
      // Tag nodes in an outer circle
      ...[...allTags].map((tag, index) => {
        const angle = (index / allTags.size) * Math.PI * 2
        const radius = Math.min(width, height) * 0.4 // 40% of the smaller dimension
        return {
          id: `tag-${tag}`,
          title: tag,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          radius: 10,
          color: "#6366f1", // Indigo for tags
          type: "tag" as const,
        }
      }),
    ]

    // Create links between entries and their tags
    const newLinks: Link[] = []
    knowledge.forEach((entry) => {
      entry.tags.forEach((tag) => {
        newLinks.push({
          source: entry.id,
          target: `tag-${tag}`,
          strength: 0.5,
        })
      })
    })

    // Create links between entries that share tags
    knowledge.forEach((entry1, i) => {
      knowledge.slice(i + 1).forEach((entry2) => {
        const sharedTags = entry1.tags.filter((tag) => entry2.tags.includes(tag))
        if (sharedTags.length > 0) {
          newLinks.push({
            source: entry1.id,
            target: entry2.id,
            strength: 0.3 * sharedTags.length,
          })
        }
      })
    })

    setNodes(newNodes)
    setLinks(newLinks)

    // Update debug info
    setDebugInfo((prev) => ({
      ...prev,
      nodeCount: newNodes.length,
      width,
      height,
    }))
  }, [knowledge])

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

      // Update debug info
      setDebugInfo((prev) => ({
        ...prev,
        width,
        height,
      }))
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Apply zoom and pan
    ctx.save()
    ctx.translate(offset.x, offset.y)
    ctx.scale(zoomLevel, zoomLevel)

    // Draw links
    ctx.strokeStyle = "rgba(147, 51, 234, 0.3)"
    ctx.lineWidth = 1 / zoomLevel

    for (const link of links) {
      const source = nodes.find((n) => n.id === link.source)
      const target = nodes.find((n) => n.id === link.target)

      if (source && target) {
        ctx.beginPath()
        ctx.moveTo(source.x, source.y)
        ctx.lineTo(target.x, target.y)
        ctx.stroke()
      }
    }

    // Draw nodes
    for (const node of nodes) {
      // Ensure radius is positive
      const radius = Math.max(node.radius, 1) // Minimum radius of 1

      ctx.beginPath()
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)

      // Highlight selected node
      if (selectedEntry && node.id === selectedEntry.id) {
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

      ctx.fill()

      // Draw node labels
      if (node === hoveredNode || (selectedEntry && node.id === selectedEntry.id) || zoomLevel > 1.5) {
        ctx.fillStyle = "#ffffff"
        ctx.font = `${Math.max(12 / zoomLevel, 8)}px Arial` // Ensure minimum font size
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"

        // Draw text with background for better readability
        const textWidth = ctx.measureText(node.title).width
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
        ctx.fillRect(node.x - textWidth / 2 - 4, node.y + radius + 4, textWidth + 8, 16 / zoomLevel)

        ctx.fillStyle = "#ffffff"
        ctx.fillText(node.title, node.x, node.y + radius + 12 / zoomLevel)
      }
    }

    ctx.restore()
  }, [nodes, links, hoveredNode, zoomLevel, offset, selectedEntry])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return

      canvas.width = container.clientWidth
      canvas.height = container.clientHeight

      // Update debug info
      setDebugInfo((prev) => ({
        ...prev,
        width: container.clientWidth,
        height: container.clientHeight,
      }))

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

  // Handle mouse interactions
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
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

    // Handle dragging
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      })
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
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

    if (clicked) {
      setSelectedNode(clicked)
      if (clicked.type === "entry") {
        const entry = knowledge.find((k) => k.id === clicked.id)
        if (entry) {
          onSelectEntry(entry)
        }
      }
    } else {
      // Start dragging the canvas
      setIsDragging(true)
      setDragStart({
        x: e.clientX - offset.x,
        y: e.clientY - offset.y,
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(prev + 0.2, 3))
  }

  const handleZoomOut = () => {
    setZoomLevel((prev) => Math.max(prev - 0.2, 0.5))
  }

  const handleReset = () => {
    setZoomLevel(1)
    setOffset({ x: 0, y: 0 })
  }

  const toggleDebug = () => {
    setShowDebug((prev) => !prev)
  }

  return (
    <Card className="h-full">
      <CardContent className="p-0 h-full flex flex-col">
        <div className="p-3 border-b border-purple-800 flex justify-between items-center">
          <div className="text-sm text-purple-300">
            {knowledge.length} entries, {new Set(knowledge.flatMap((k) => k.tags)).size} tags
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={toggleDebug}
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              Debug
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleZoomOut}
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <ZoomOut size={14} />
            </Button>
            <span className="flex items-center text-xs text-white px-1">{Math.round(zoomLevel * 100)}%</span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleZoomIn}
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              <ZoomIn size={14} />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              className="bg-purple-900/50 border-purple-500 text-white hover:bg-purple-800 hover:text-white"
            >
              Reset
            </Button>
          </div>
        </div>

        <div ref={containerRef} className="flex-1 relative bg-black">
          <canvas
            ref={canvasRef}
            className="absolute inset-0 cursor-grab"
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />

          {showDebug && (
            <div className="absolute top-4 left-4 bg-black/80 text-white p-3 rounded-md text-xs font-mono z-10">
              <div>
                Canvas: {debugInfo.width}x{debugInfo.height}
              </div>
              <div>Nodes: {debugInfo.nodeCount}</div>
              <div>Zoom: {zoomLevel.toFixed(2)}</div>
              <div>
                Offset: {offset.x.toFixed(0)},{offset.y.toFixed(0)}
              </div>
              <div>Hovered: {hoveredNode?.title || "none"}</div>
            </div>
          )}
        </div>

        {selectedNode && selectedNode.type === "tag" && (
          <div className="absolute bottom-4 left-4 bg-purple-950/80 backdrop-blur-sm p-3 rounded-lg border border-purple-700 shadow-md max-w-xs">
            <h3 className="font-medium text-white mb-1">Tag: {selectedNode.title.replace("tag-", "")}</h3>
            <ScrollArea className="h-32">
              <div className="space-y-1">
                {knowledge
                  .filter((entry) => entry.tags.includes(selectedNode.title.replace("tag-", "")))
                  .map((entry) => (
                    <div
                      key={entry.id}
                      className="p-2 text-sm rounded hover:bg-purple-800/50 cursor-pointer text-white"
                      onClick={() => onSelectEntry(entry)}
                    >
                      {entry.title}
                    </div>
                  ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
