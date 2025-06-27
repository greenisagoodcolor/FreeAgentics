"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Search,
  ShoppingCart,
  BookOpen,
  Shield,
  Brain,
  Plus,
  Sparkles,
} from "lucide-react";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import {
  createAgent,
  quickStartAgents,
  AgentTemplate,
} from "@/store/slices/agentSlice";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Slider } from "@/components/ui/slider";

const iconMap = {
  Search,
  ShoppingCart,
  BookOpen,
  Shield,
  Brain,
};

const categoryColors = {
  researcher: "#10B981",
  student: "#8B5CF6",
  expert: "#3B82F6",
  generalist: "#F59E0B",
  contrarian: "#EF4444",
};

interface AgentCreationModalProps {
  template: AgentTemplate;
  isOpen: boolean;
  onClose: () => void;
}

const AgentCreationModal: React.FC<AgentCreationModalProps> = ({
  template,
  isOpen,
  onClose,
}) => {
  const dispatch = useAppDispatch();
  const [agentName, setAgentName] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [parameters, setParameters] = useState(template.defaultParameters);

  const generateSuggestedNames = (template: AgentTemplate) => {
    const adjectives = ["Swift", "Wise", "Bold", "Keen", "Sharp"];
    const suffixes = ["Alpha", "Beta", "Prime", "Neo", "Core"];
    return [
      `${template.name} ${Math.floor(Math.random() * 100)}`,
      `${adjectives[Math.floor(Math.random() * adjectives.length)]} ${template.name}`,
      `${template.name} ${suffixes[Math.floor(Math.random() * suffixes.length)]}`,
    ];
  };

  const handleCreate = () => {
    dispatch(
      createAgent({
        templateId: template.id,
        name: agentName || `${template.name} 1`,
        parameterOverrides: showAdvanced ? parameters : undefined,
      }),
    );
    onClose();
    setAgentName("");
    setShowAdvanced(false);
    setParameters(template.defaultParameters);
  };

  const IconComponent = iconMap[template.icon as keyof typeof iconMap];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-md bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]">
        <DialogHeader>
          <DialogTitle className="font-ui text-[var(--text-primary)] flex items-center gap-2">
            <IconComponent
              className="w-5 h-5"
              style={{ color: template.color }}
            />
            Create {template.name}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Template Preview */}
          <motion.div
            className="agent-card mx-auto relative"
            style={{
              backgroundColor: template.color + "20",
              borderColor: template.color,
            }}
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <div className="p-2 text-center">
              <div
                className="w-12 h-12 mx-auto mb-2 rounded-full flex items-center justify-center"
                style={{ backgroundColor: template.color }}
              >
                <IconComponent className="w-6 h-6 text-white" />
              </div>
              <div className="text-xs font-medium text-[var(--text-primary)] truncate">
                {agentName || `${template.name} 1`}
              </div>
              <Badge
                variant="secondary"
                className="text-xs mt-1"
                style={{
                  backgroundColor: template.color + "30",
                  color: template.color,
                }}
              >
                {template.category}
              </Badge>
            </div>
          </motion.div>

          {/* Agent Name */}
          <div className="space-y-2">
            <Label
              htmlFor="agentName"
              className="font-ui text-sm text-[var(--text-primary)]"
            >
              Agent Name
            </Label>
            <Input
              id="agentName"
              value={agentName}
              onChange={(e) => setAgentName(e.target.value)}
              placeholder={`${template.name} 1`}
              className="bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)] text-[var(--text-primary)]"
            />
            <div className="flex gap-1 flex-wrap">
              {generateSuggestedNames(template).map((name, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => setAgentName(name)}
                  className="text-xs h-6 px-2 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)] hover:bg-[var(--accent-primary)]"
                >
                  {name}
                </Button>
              ))}
            </div>
          </div>

          {/* Advanced Settings */}
          <Accordion type="single" collapsible>
            <AccordionItem
              value="advanced"
              className="border-[var(--bg-tertiary)]"
            >
              <AccordionTrigger
                className="text-[var(--text-primary)] hover:text-[var(--accent-primary)]"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                Advanced Settings
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <Label className="text-sm text-[var(--text-secondary)]">
                      Response Threshold:{" "}
                      {parameters.responseThreshold.toFixed(2)}
                    </Label>
                    <Slider
                      value={[parameters.responseThreshold]}
                      onValueChange={([value]) =>
                        setParameters((prev) => ({
                          ...prev,
                          responseThreshold: value,
                        }))
                      }
                      max={1}
                      min={0}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label className="text-sm text-[var(--text-secondary)]">
                      Turn-taking Probability:{" "}
                      {parameters.turnTakingProbability.toFixed(2)}
                    </Label>
                    <Slider
                      value={[parameters.turnTakingProbability]}
                      onValueChange={([value]) =>
                        setParameters((prev) => ({
                          ...prev,
                          turnTakingProbability: value,
                        }))
                      }
                      max={1}
                      min={0}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <Label className="text-sm text-[var(--text-secondary)]">
                      Conversation Engagement:{" "}
                      {parameters.conversationEngagement.toFixed(2)}
                    </Label>
                    <Slider
                      value={[parameters.conversationEngagement]}
                      onValueChange={([value]) =>
                        setParameters((prev) => ({
                          ...prev,
                          conversationEngagement: value,
                        }))
                      }
                      max={1}
                      min={0}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={onClose}
              variant="outline"
              className="flex-1 bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)] hover:bg-[var(--bg-tertiary)]"
            >
              Cancel
            </Button>
            <Button onClick={handleCreate} className="flex-1 btn-primary">
              Create Agent
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AgentTemplateSelector: React.FC = () => {
  const dispatch = useAppDispatch();
  const templates = useAppSelector((state) => state.agents.templates);
  const [selectedTemplate, setSelectedTemplate] =
    useState<AgentTemplate | null>(null);

  const handleQuickStart = () => {
    dispatch(quickStartAgents());
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-ui text-xl font-semibold text-[var(--text-primary)]">
            Agent Templates
          </h2>
          <p className="font-ui text-sm text-[var(--text-secondary)] mt-1">
            Create agents using pre-configured templates
          </p>
        </div>
        <Button
          onClick={handleQuickStart}
          className="btn-primary flex items-center gap-2"
        >
          <Sparkles className="w-4 h-4" />
          Quick Start
        </Button>
      </div>

      {/* Template Selector - Horizontal Scrollable */}
      <div className="relative">
        <div
          className="flex gap-4 overflow-x-auto pb-4 scroll-smooth"
          style={{
            scrollSnapType: "x mandatory",
            scrollbarWidth: "thin",
            scrollbarColor: "var(--bg-tertiary) transparent",
          }}
        >
          {Object.values(templates).map((template) => {
            const IconComponent =
              iconMap[template.icon as keyof typeof iconMap];

            return (
              <motion.div
                key={template.id}
                className="agent-card flex-shrink-0 cursor-pointer relative overflow-hidden"
                style={{ scrollSnapAlign: "start" }}
                whileHover={{ y: -4, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedTemplate(template)}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              >
                {/* Card Content */}
                <div className="p-3 h-full flex flex-col items-center justify-between">
                  {/* Icon */}
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center mb-2"
                    style={{ backgroundColor: template.color }}
                  >
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>

                  {/* Name */}
                  <div className="text-center flex-1">
                    <div className="font-ui text-sm font-medium text-[var(--text-primary)] mb-1">
                      {template.name}
                    </div>
                    <Badge
                      variant="secondary"
                      className="text-xs"
                      style={{
                        backgroundColor: template.color + "20",
                        color: template.color,
                        border: `1px solid ${template.color}40`,
                      }}
                    >
                      {template.category}
                    </Badge>
                  </div>

                  {/* Add Button */}
                  <motion.div
                    className="mt-2"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center border-2"
                      style={{
                        borderColor: template.color,
                        backgroundColor: template.color + "20",
                      }}
                    >
                      <Plus
                        className="w-3 h-3"
                        style={{ color: template.color }}
                      />
                    </div>
                  </motion.div>
                </div>

                {/* Hover Overlay */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 flex items-end p-2"
                  whileHover={{ opacity: 1 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="text-xs text-white text-center w-full">
                    {template.defaultBiography.slice(0, 60)}...
                  </div>
                </motion.div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Template Details */}
      <div className="grid grid-cols-1 gap-4">
        {Object.values(templates).map((template) => {
          const IconComponent = iconMap[template.icon as keyof typeof iconMap];

          return (
            <Card
              key={template.id}
              className="bg-[var(--bg-secondary)] border-[var(--bg-tertiary)]"
            >
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-3 text-[var(--text-primary)]">
                  <IconComponent
                    className="w-5 h-5"
                    style={{ color: template.color }}
                  />
                  {template.name}
                  <Badge
                    variant="secondary"
                    style={{
                      backgroundColor: template.color + "20",
                      color: template.color,
                    }}
                  >
                    {template.category}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                  {template.defaultBiography}
                </p>
                <div className="flex flex-wrap gap-1">
                  {template.defaultKnowledgeDomains.map((domain, index) => (
                    <Badge
                      key={index}
                      variant="outline"
                      className="text-xs bg-[var(--bg-tertiary)] border-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                    >
                      {domain}
                    </Badge>
                  ))}
                </div>
                <div className="flex justify-end">
                  <Button
                    onClick={() => setSelectedTemplate(template)}
                    size="sm"
                    className="btn-primary"
                  >
                    Create {template.name}
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Agent Creation Modal */}
      {selectedTemplate && (
        <AgentCreationModal
          template={selectedTemplate}
          isOpen={!!selectedTemplate}
          onClose={() => setSelectedTemplate(null)}
        />
      )}
    </div>
  );
};

export default AgentTemplateSelector;
