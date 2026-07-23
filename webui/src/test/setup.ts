import "@testing-library/jest-dom/vitest";

// Under Node's experimental web storage, the global `localStorage` is present
// but unusable without a backing file, which shadows jsdom's. Install a
// deterministic in-memory Storage so hooks that persist to localStorage are
// testable in isolation.
class MemoryStorage implements Storage {
  private store = new Map<string, string>();
  get length(): number {
    return this.store.size;
  }
  clear(): void {
    this.store.clear();
  }
  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }
  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null;
  }
  removeItem(key: string): void {
    this.store.delete(key);
  }
  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }
}

const storage = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", {
  value: storage,
  configurable: true,
  writable: true,
});
if (typeof window !== "undefined") {
  Object.defineProperty(window, "localStorage", {
    value: storage,
    configurable: true,
    writable: true,
  });
}
