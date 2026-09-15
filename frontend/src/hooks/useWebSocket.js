import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_BASE } from '../utils/constants';

/**
 * WebSocket hook for real-time pipeline progress streaming.
 * Auto-reconnects with exponential backoff on disconnect.
 */
export function useWebSocket(jobId) {
  const [messages, setMessages] = useState([]);
  const [lastMessage, setLastMessage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimeout = useRef(null);
  const reconnectDelay = useRef(1000);

  const connect = useCallback(() => {
    if (!jobId) return;

    const url = `${WS_BASE}/ws/jobs/${jobId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectDelay.current = 1000; // Reset backoff
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        setMessages(prev => [...prev, data]);
      } catch {
        // Non-JSON message, ignore
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect with exponential backoff (max 30s)
      reconnectTimeout.current = setTimeout(() => {
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
        connect();
      }, reconnectDelay.current);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [jobId]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, [connect]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, lastMessage, isConnected, clearMessages };
}
