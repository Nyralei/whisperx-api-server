// FastAPI serves interactive API docs at /docs (Swagger UI) and /redoc (ReDoc)
// on this same origin — absolute paths so they resolve from under /webui/.
const linkClass =
  "rounded-sm px-2 py-1 text-xs font-medium text-fg-muted hover:bg-surface-2 hover:text-fg " +
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export function DocsLinks() {
  return (
    <nav aria-label="API documentation" className="flex items-center gap-0.5">
      <a
        href="/docs"
        target="_blank"
        rel="noopener noreferrer"
        title="Interactive API reference (Swagger UI)"
        className={linkClass}
      >
        Swagger
      </a>
      <a
        href="/redoc"
        target="_blank"
        rel="noopener noreferrer"
        title="API reference (ReDoc)"
        className={linkClass}
      >
        ReDoc
      </a>
    </nav>
  );
}
