"use client";

import React, { useState, useRef } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/components/ui/use-toast";
import {
  Upload,
  File,
  CheckCircle,
  AlertCircle,
  Loader2,
  Info,
  FileJson,
  Users,
  MessageSquare,
  Database,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ImportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImportComplete?: (importId: string) => void;
}

interface ImportMetadata {
  id: string;
  name: string;
  description: string;
  version: string;
  createdAt: string;
  createdBy: string;
  components: string[];
  statistics: {
    totalAgents: number;
    totalConversations: number;
    totalMessages: number;
    totalKnowledgeNodes: number;
  };
}

export function ExperimentImportModal({
  open,
  onOpenChange,
  onImportComplete,
}: ImportModalProps) {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState<
    "upload" | "preview" | "importing" | "complete" | "error"
  >("upload");
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [importMetadata, setImportMetadata] = useState<ImportMetadata | null>(
    null,
  );
  const [importId, setImportId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      setSelectedFile(file);

      // Simulate reading the file metadata
      setTimeout(() => {
        // Mock metadata that would be extracted from the file
        const mockMetadata: ImportMetadata = {
          id: `exp_${Math.random().toString(36).substring(2, 10)}`,
          name: file.name.replace(/\.[^/.]+$/, ""),
          description:
            "Experiment state export containing agents, conversations, and knowledge graphs.",
          version: "1.0.0",
          createdAt: new Date().toISOString(),
          createdBy: "user@example.com",
          components: [
            "Agents",
            "Conversations",
            "Knowledge Graphs",
            "Parameters",
          ],
          statistics: {
            totalAgents: Math.floor(Math.random() * 10) + 1,
            totalConversations: Math.floor(Math.random() * 20) + 1,
            totalMessages: Math.floor(Math.random() * 100) + 10,
            totalKnowledgeNodes: Math.floor(Math.random() * 50) + 5,
          },
        };

        setImportMetadata(mockMetadata);
        setStep("preview");
      }, 1000);
    }
  };

  const handleImport = async () => {
    if (!selectedFile || !importMetadata) return;

    setStep("importing");
    setProgress(0);

    // Simulate import process with progress updates
    const interval = setInterval(() => {
      setProgress((prev) => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          // Simulate API call completion
          setTimeout(() => {
            if (Math.random() > 0.1) {
              // 90% success rate
              setImportId(importMetadata.id);
              setStep("complete");
              if (onImportComplete) {
                onImportComplete(importMetadata.id);
              }
            } else {
              setErrorMessage(
                "Failed to validate experiment data. The file may be corrupted or incompatible.",
              );
              setStep("error");
            }
          }, 500);
          return 100;
        }
        return newProgress;
      });
    }, 300);
  };

  const handleClose = () => {
    // Reset state when closing
    if (step === "complete" || step === "error") {
      setTimeout(() => {
        setStep("upload");
        setProgress(0);
        setSelectedFile(null);
        setImportMetadata(null);
        setImportId(null);
        setErrorMessage("");
      }, 300);
    }
    onOpenChange(false);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const renderUploadStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Import Experiment State</DialogTitle>
        <DialogDescription>
          Import a previously exported experiment state to recreate your
          research environment.
        </DialogDescription>
      </DialogHeader>

      <div className="py-8">
        <div
          className="border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={triggerFileInput}
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".json,.gz,.zip,.tar.gz"
            onChange={handleFileChange}
          />
          <Upload className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">Upload Experiment File</h3>
          <p className="text-sm text-muted-foreground text-center mb-4">
            Drag and drop your experiment file here, or click to browse
          </p>
          <Button>
            <File className="mr-2 h-4 w-4" />
            Select File
          </Button>
          <p className="text-xs text-muted-foreground mt-4">
            Supported formats: .json, .gz, .zip, .tar.gz
          </p>
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Cancel
        </Button>
      </DialogFooter>
    </>
  );

  const renderPreviewStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Import Preview</DialogTitle>
        <DialogDescription>
          Review the experiment data before importing.
        </DialogDescription>
      </DialogHeader>

      <div className="py-4 space-y-6">
        <div className="flex items-start space-x-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <FileJson className="h-8 w-8 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-medium">{selectedFile?.name}</h3>
            <p className="text-sm text-muted-foreground">
              {(selectedFile?.size &&
                (selectedFile.size / (1024 * 1024)).toFixed(2)) ||
                0}{" "}
              MB •{" "}
              {importMetadata?.createdAt
                ? new Date(importMetadata.createdAt).toLocaleDateString()
                : "Unknown date"}
            </p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Experiment Information</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-1">
                <span className="text-muted-foreground">Name:</span>
                <span className="font-medium col-span-2">
                  {importMetadata?.name}
                </span>
              </div>
              {importMetadata?.description && (
                <div className="grid grid-cols-3 gap-1">
                  <span className="text-muted-foreground">Description:</span>
                  <span className="col-span-2">
                    {importMetadata.description}
                  </span>
                </div>
              )}
              <div className="grid grid-cols-3 gap-1">
                <span className="text-muted-foreground">Version:</span>
                <span>{importMetadata?.version}</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                <span className="text-muted-foreground">Created By:</span>
                <span>{importMetadata?.createdBy}</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                <span className="text-muted-foreground">Components:</span>
                <span className="col-span-2">
                  {importMetadata?.components.join(", ")}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-3 gap-4">
          <Card className="p-4 flex flex-col items-center">
            <Users className="h-6 w-6 text-blue-500 mb-2" />
            <div className="text-2xl font-bold">
              {importMetadata?.statistics.totalAgents}
            </div>
            <div className="text-xs text-muted-foreground">Agents</div>
          </Card>
          <Card className="p-4 flex flex-col items-center">
            <MessageSquare className="h-6 w-6 text-green-500 mb-2" />
            <div className="text-2xl font-bold">
              {importMetadata?.statistics.totalConversations}
            </div>
            <div className="text-xs text-muted-foreground">Conversations</div>
          </Card>
          <Card className="p-4 flex flex-col items-center">
            <Database className="h-6 w-6 text-purple-500 mb-2" />
            <div className="text-2xl font-bold">
              {importMetadata?.statistics.totalKnowledgeNodes}
            </div>
            <div className="text-xs text-muted-foreground">Knowledge Nodes</div>
          </Card>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-md p-3 flex items-start space-x-3">
          <Info className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-800">
            <p className="font-medium">Import Notice</p>
            <p className="mt-1">
              Importing this experiment will create new agents, conversations,
              and knowledge graphs in your environment. Existing data will not
              be overwritten.
            </p>
          </div>
        </div>
      </div>

      <DialogFooter className="space-x-2">
        <Button variant="outline" onClick={() => setStep("upload")}>
          Back
        </Button>
        <Button onClick={handleImport}>
          <Upload className="mr-2 h-4 w-4" />
          Import Experiment
        </Button>
      </DialogFooter>
    </>
  );

  const renderImportingStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Importing Experiment</DialogTitle>
        <DialogDescription>
          Please wait while we import your experiment data...
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative w-20 h-20 flex items-center justify-center">
            <div className="absolute inset-0 border-4 border-primary/30 rounded-full" />
            <div
              className="absolute inset-0 border-4 border-primary rounded-full"
              style={{
                clipPath: `polygon(0% 0%, ${progress}% 0%, ${progress}% 100%, 0% 100%)`,
                transition: "clip-path 0.3s ease-in-out",
              }}
            />
            <span className="text-lg font-semibold">
              {Math.round(progress)}%
            </span>
          </div>
        </div>

        <div className="space-y-2">
          <Progress value={progress} className="h-2" />
          <div className="text-center text-sm text-muted-foreground">
            {progress < 30 && "Validating experiment data..."}
            {progress >= 30 && progress < 60 && "Preparing database..."}
            {progress >= 60 && progress < 90 && "Importing components..."}
            {progress >= 90 && "Finalizing import..."}
          </div>
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" disabled>
          Cancel
        </Button>
        <Button disabled>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Importing...
        </Button>
      </DialogFooter>
    </>
  );

  const renderCompleteStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Import Complete</DialogTitle>
        <DialogDescription>
          Your experiment has been successfully imported.
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-3 rounded-full bg-green-100">
            <CheckCircle className="h-10 w-10 text-green-600" />
          </div>
          <div className="text-center">
            <h3 className="text-lg font-medium">Import Successful</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Import ID: {importId}
            </p>
          </div>
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Import Summary</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-3">
            <div className="grid grid-cols-2 gap-1">
              <span className="text-muted-foreground">Agents:</span>
              <span className="font-medium">
                {importMetadata?.statistics.totalAgents} imported
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1">
              <span className="text-muted-foreground">Conversations:</span>
              <span className="font-medium">
                {importMetadata?.statistics.totalConversations} imported
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1">
              <span className="text-muted-foreground">Knowledge Nodes:</span>
              <span className="font-medium">
                {importMetadata?.statistics.totalKnowledgeNodes} imported
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1">
              <span className="text-muted-foreground">Import Date:</span>
              <span className="font-medium">{new Date().toLocaleString()}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Close
        </Button>
        <Button onClick={() => (window.location.href = "/experiments")}>
          View Experiment
        </Button>
      </DialogFooter>
    </>
  );

  const renderErrorStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Import Failed</DialogTitle>
        <DialogDescription>
          There was an error importing your experiment.
        </DialogDescription>
      </DialogHeader>

      <div className="py-8 space-y-6">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-3 rounded-full bg-red-100">
            <AlertCircle className="h-10 w-10 text-red-600" />
          </div>
          <div className="text-center">
            <h3 className="text-lg font-medium">Import Failed</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {errorMessage || "An unexpected error occurred during import."}
            </p>
          </div>
        </div>

        <Card className="bg-red-50 border-red-200">
          <CardContent className="text-sm p-4">
            <h4 className="font-medium text-red-800 mb-2">
              Troubleshooting Tips
            </h4>
            <ul className="list-disc pl-5 space-y-1 text-red-700">
              <li>Check that the file format is supported</li>
              <li>Ensure the export was created with a compatible version</li>
              <li>Verify the file is not corrupted</li>
              <li>Try re-exporting the experiment from the source system</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Close
        </Button>
        <Button onClick={() => setStep("upload")}>Try Again</Button>
      </DialogFooter>
    </>
  );

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        {step === "upload" && renderUploadStep()}
        {step === "preview" && renderPreviewStep()}
        {step === "importing" && renderImportingStep()}
        {step === "complete" && renderCompleteStep()}
        {step === "error" && renderErrorStep()}
      </DialogContent>
    </Dialog>
  );
}
