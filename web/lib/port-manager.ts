// web/lib/port-manager.ts
import { exec } from "child_process";
import { createServer } from "net";
import { promisify } from "util";

const execAsync = promisify(exec);

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = createServer();

    server.listen(port, () => {
      server.close(() => resolve(true));
    });

    server.on("error", () => resolve(false));
  });
}

export async function findAvailablePort(
  startPort: number,
  portRange: number[] = [3000, 3001, 3002],
): Promise<number> {
  for (const port of portRange) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }

  // If no port in range is available, try sequential search
  let port = Math.max(...portRange) + 1;
  while (port < 65535) {
    if (await isPortAvailable(port)) {
      return port;
    }
    port++;
  }

  throw new Error("No available ports found");
}

export async function killProcessOnPort(port: number): Promise<void> {
  try {
    await execAsync(`lsof -ti:${port} | xargs -r kill`);
    console.log(`Killed processes on port ${port}`);
  } catch (error) {
    // Port might not have any processes - this is fine
    console.log(
      `No processes found on port ${port} or kill failed:`,
      error instanceof Error ? error.message : String(error),
    );
  }
}

export interface ServerStartOptions {
  preferredPort?: number;
}

export interface ServerStartResult {
  port: number;
  status: string;
}

export async function startServer(options: ServerStartOptions): Promise<ServerStartResult> {
  // Check for PORT environment variable
  const envPort = process.env.PORT ? parseInt(process.env.PORT, 10) : null;
  const preferredPort = envPort || options.preferredPort || 3000;

  // Try to kill any existing processes on preferred port
  try {
    await killProcessOnPort(preferredPort);
  } catch (error) {
    console.warn(
      `Failed to clean up port ${preferredPort}:`,
      error instanceof Error ? error.message : String(error),
    );
  }

  // Find available port starting from preferred port
  const availablePort = await findAvailablePort(preferredPort, [
    preferredPort,
    preferredPort + 1,
    preferredPort + 2,
  ]);

  return {
    port: availablePort,
    status: "started",
  };
}
