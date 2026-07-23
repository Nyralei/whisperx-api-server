import { useState } from "react";

import type { ServerConnection } from "../hooks/useServerConnection";
import { Button, Field, inputClass, Toggle } from "./ui";

export function ApiKeyGate(props: { connection: ServerConnection }) {
  const { connection } = props;
  const [key, setKey] = useState("");
  const [remember, setRemember] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  return (
    <form
      className="flex flex-col gap-3 rounded-sm border border-warn-border bg-warn-bg p-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (!key.trim()) return;
        setSubmitting(true);
        void connection.submitKey(key.trim(), remember).finally(() => {
          setSubmitting(false);
        });
      }}
    >
      <p className="text-sm text-warn-strong">This server requires an API key.</p>
      <Field id="api-key" label="API key">
        <input
          id="api-key"
          type="password"
          autoComplete="off"
          className={inputClass}
          value={key}
          onChange={(e) => setKey(e.target.value)}
        />
      </Field>
      <Toggle
        id="api-key-remember"
        label="Remember in this browser"
        checked={remember}
        onChange={setRemember}
      />
      {connection.keyError ? (
        <p role="alert" className="text-sm text-danger-text">
          {connection.keyError}
        </p>
      ) : null}
      <Button type="submit" disabled={submitting || !key.trim()}>
        {submitting ? "Checking…" : "Connect"}
      </Button>
    </form>
  );
}
