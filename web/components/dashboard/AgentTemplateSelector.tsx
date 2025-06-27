'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Search,
  ShoppingCart,
  BookOpen,
  Shield,
  Brain,
  Plus,
  Sparkles,
} from 'lucide-react';
import { useAppDispatch, useAppSelector } from '@/store';
import { createAgent, AgentTemplate } from '@/store/slices/agentSlice';
import { cn } from '@/lib/utils';

const iconMap = {
  Search,
  ShoppingCart,
  BookOpen,
  Shield,
  Brain,
};

interface AgentTemplateSelectorProps {
  onTemplateSelect?: (template: AgentTemplate) => void;
  className?: string;
}

export function AgentTemplateSelector({ onTemplateSelect, className }: AgentTemplateSelectorProps) {
  const dispatch = useAppDispatch();
  const templates = useAppSelector(state => Object.values(state.agents.templates));
  const agents = useAppSelector(state => state.agents.agents);

  const handleQuickCreate = (template: AgentTemplate) => {
    dispatch(createAgent({ templateId: template.id }));
  };

  const handleTemplateClick = (template: AgentTemplate) => {
    if (onTemplateSelect) {
      onTemplateSelect(template);
    } else {
      handleQuickCreate(template);
    }
  };

  const getAgentCount = (templateId: string) => {
    return Object.values(agents).filter(agent => agent.templateId === templateId).length;
  };

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-300">AGENT TEMPLATES</h3>
        <Button
          variant="ghost"
          size="sm"
          className="text-xs gap-1"
          onClick={() => {
            // Quick start - create 3 default agents
            dispatch(createAgent({ templateId: 'explorer' }));
            dispatch(createAgent({ templateId: 'scholar' }));
            dispatch(createAgent({ templateId: 'merchant' }));
          }}
        >
          <Sparkles className="h-3 w-3" />
          Quick Start
        </Button>
      </div>

      <div 
        className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-zinc-800 scrollbar-thumb-zinc-600"
        style={{ scrollSnapType: 'x mandatory' }}
      >
        {templates.map((template) => {
          const Icon = iconMap[template.icon as keyof typeof iconMap] || Brain;
          const count = getAgentCount(template.id);

          return (
            <motion.div
              key={template.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              style={{ scrollSnapAlign: 'start' }}
            >
              <Card
                className={cn(
                  "min-w-[240px] bg-zinc-900 border-zinc-700 hover:border-zinc-600",
                  "cursor-pointer transition-all duration-200 p-4",
                  "hover:shadow-lg hover:shadow-black/20"
                )}
                onClick={() => handleTemplateClick(template)}
              >
                <div className="flex items-start gap-3">
                  <div
                    className="p-2.5 rounded-lg"
                    style={{ backgroundColor: template.color + '20' }}
                  >
                    <Icon className="h-5 w-5" style={{ color: template.color }} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="font-medium text-sm">{template.name}</h4>
                      <Badge 
                        variant="outline" 
                        className="text-xs"
                        style={{ borderColor: template.color + '40', color: template.color }}
                      >
                        {template.category}
                      </Badge>
                    </div>
                    
                    <p className="text-xs text-zinc-400 line-clamp-2 mb-2">
                      {template.defaultBiography}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex flex-wrap gap-1">
                        {template.defaultKnowledgeDomains.slice(0, 2).map((domain) => (
                          <span
                            key={domain}
                            className="text-xs px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400"
                          >
                            {domain}
                          </span>
                        ))}
                        {template.defaultKnowledgeDomains.length > 2 && (
                          <span className="text-xs text-zinc-500">
                            +{template.defaultKnowledgeDomains.length - 2}
                          </span>
                        )}
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {count > 0 && (
                          <span className="text-xs text-zinc-500">{count} active</span>
                        )}
                        <Plus className="h-3 w-3 text-zinc-500" />
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Quick metrics */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="text-center p-2 bg-zinc-900 rounded">
          <div className="text-zinc-400">Total Agents</div>
          <div className="font-mono text-sm">{Object.keys(agents).length}</div>
        </div>
        <div className="text-center p-2 bg-zinc-900 rounded">
          <div className="text-zinc-400">Active</div>
          <div className="font-mono text-sm text-green-500">
            {Object.values(agents).filter(a => a.status === 'active').length}
          </div>
        </div>
        <div className="text-center p-2 bg-zinc-900 rounded">
          <div className="text-zinc-400">Typing</div>
          <div className="font-mono text-sm text-yellow-500">
            {Object.values(agents).filter(a => a.status === 'typing').length}
          </div>
        </div>
      </div>
    </div>
  );
} 