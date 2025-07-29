import { ensureDevToken } from './auth';

let ws: WebSocket | null = null;

export async function getSocket() {
  if (ws && ws.readyState <= 1) return ws;          // OPEN | CONNECTING
  const token = await ensureDevToken();
  if (!token) {
    throw new Error('No authentication token available');
  }
  const url = new URL('/api/v1/ws/dev', location.origin.replace('http', 'ws'));
  url.searchParams.set('token', token);
  ws = new WebSocket(url.toString());
  return ws;
}