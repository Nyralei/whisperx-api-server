/**
 * Reachability + auth handshake. GET /info doubles as the auth probe: it sits
 * behind the same key dependency as the rest of the API, so an unauthenticated
 * 200 means no key is configured (the key field is then hidden), while 401/403
 * means a key is required.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getInfo } from "../api/client";
import type { ServerInfo } from "../api/types";

const STORAGE_KEY = "whisperx-webui-api-key";

export type ConnectionPhase = "connecting" | "unreachable" | "need-key" | "ready";

export interface ServerConnection {
  phase: ConnectionPhase;
  authRequired: boolean;
  info: ServerInfo | null;
  apiKey: string | null;
  keyError: string | null;
  submitKey: (key: string, remember: boolean) => Promise<void>;
  clearKey: () => void;
  retry: () => void;
}

export function useServerConnection(): ServerConnection {
  const [phase, setPhase] = useState<ConnectionPhase>("connecting");
  const [authRequired, setAuthRequired] = useState(false);
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [keyError, setKeyError] = useState<string | null>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  const connect = useCallback(async () => {
    setPhase("connecting");
    setKeyError(null);

    const anonymous = await getInfo(null);
    if (!aliveRef.current) return;

    if (anonymous.ok) {
      setAuthRequired(false);
      setApiKey(null);
      setInfo(anonymous.info);
      setPhase("ready");
      return;
    }
    if (anonymous.status === null) {
      setPhase("unreachable");
      return;
    }
    if (anonymous.status !== 401 && anonymous.status !== 403) {
      // /info failing with an unexpected status: treat as unreachable.
      setPhase("unreachable");
      return;
    }

    setAuthRequired(true);
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const withKey = await getInfo(stored);
      if (!aliveRef.current) return;
      if (withKey.ok) {
        setApiKey(stored);
        setInfo(withKey.info);
        setPhase("ready");
        return;
      }
      localStorage.removeItem(STORAGE_KEY);
    }
    setPhase("need-key");
  }, []);

  useEffect(() => {
    void connect();
  }, [connect]);

  const submitKey = useCallback(async (key: string, remember: boolean) => {
    setKeyError(null);
    const result = await getInfo(key);
    if (!aliveRef.current) return;
    if (result.ok) {
      setApiKey(key);
      setInfo(result.info);
      setPhase("ready");
      if (remember) localStorage.setItem(STORAGE_KEY, key);
      return;
    }
    setKeyError(
      result.status === 401 || result.status === 403
        ? "Invalid API key."
        : "Could not verify the key — server unreachable.",
    );
  }, []);

  const clearKey = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setApiKey(null);
    setPhase("need-key");
  }, []);

  return {
    phase,
    authRequired,
    info,
    apiKey,
    keyError,
    submitKey,
    clearKey,
    retry: () => void connect(),
  };
}
