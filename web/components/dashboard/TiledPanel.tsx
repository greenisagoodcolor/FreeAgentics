"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Minimize2,
  Maximize2,
  X,
  Move,
  MoreVertical,
} from "lucide-react";

interface TiledPanelProps {
  id: string;
  title: string;
  gridArea?: {
    rowStart: number;
    rowEnd: number;
    colStart: number;
    colEnd: number;
  };
  closable?: boolean;
  detachable?: boolean;
  focused?: boolean;
  onFocus?: () => void;
  onClose?: () => void;
  children: React.ReactNode;
  className?: string;
}

const TiledPanel: React.FC<TiledPanelProps> = ({
  id,
  title,
  gridArea,
  closable = false,
  detachable = false,
  focused = false,
  onFocus,
  onClose,
  children,
  className = ""
}) => {
  const [isMinimized, setIsMinimized] = useState(false);

  const gridStyles = gridArea ? {
    gridRowStart: gridArea.rowStart,
    gridRowEnd: gridArea.rowEnd,
    gridColumnStart: gridArea.colStart,
    gridColumnEnd: gridArea.colEnd,
  } : {};

  return (
    <motion.div
      className={`tiled-panel ${focused ? 'focused' : ''} ${className}`}
      style={gridStyles}
      onClick={onFocus}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      layout
    >
      {/* Panel Header */}
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <h3 className="panel-title">{title}</h3>
          <div className="flex items-center gap-1">
            <div className="status-dot active"></div>
            <span className="text-xs text-[var(--text-secondary)]">LIVE</span>
          </div>
        </div>
        
        <div className="panel-controls">
          <motion.button
            className="panel-btn"
            title="Minimize"
            onClick={(e) => {
              e.stopPropagation();
              setIsMinimized(!isMinimized);
            }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Minimize2 className="w-3 h-3" />
          </motion.button>
          
          <motion.button
            className="panel-btn"
            title="Maximize"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Maximize2 className="w-3 h-3" />
          </motion.button>
          
          {detachable && (
            <motion.button
              className="panel-btn"
              title="Pop out"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <Move className="w-3 h-3" />
            </motion.button>
          )}
          
          {closable && (
            <motion.button
              className="panel-btn"
              title="Close"
              onClick={(e) => {
                e.stopPropagation();
                onClose?.();
              }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <X className="w-3 h-3" />
            </motion.button>
          )}
          
          <motion.button
            className="panel-btn"
            title="More options"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <MoreVertical className="w-3 h-3" />
          </motion.button>
        </div>
      </div>
      
      {/* Panel Content */}
      <AnimatePresence>
        {!isMinimized && (
          <motion.div
            className="panel-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default TiledPanel; 