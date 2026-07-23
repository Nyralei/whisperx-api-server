import { Component, type ErrorInfo, type ReactNode } from "react";

import { Button } from "./ui";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Catches render-time errors so a component crash shows a recoverable message
 * instead of a blank page. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Web UI crashed:", error, info);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="flex min-h-screen items-center justify-center bg-canvas p-6 text-fg">
        <div className="max-w-lg text-center">
          <h1 className="text-lg font-semibold">Something went wrong</h1>
          <p className="mt-2 text-sm text-fg-muted">
            The interface hit an unexpected error. Reloading usually fixes it.
          </p>
          <pre className="mt-3 overflow-auto rounded-md bg-surface-2 p-3 text-left text-xs text-danger-text">
            {this.state.error.message}
          </pre>
          <Button className="mt-4" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </div>
      </div>
    );
  }
}
