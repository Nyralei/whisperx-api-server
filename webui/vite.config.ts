/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev-server proxy target: a running whisperx-api-server instance.
const API_TARGET = process.env.WEBUI_DEV_API ?? "http://localhost:8000";

export default defineConfig({
  // Assets are served by FastAPI under the /webui mount.
  base: "/webui/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/v1": API_TARGET,
      "/info": API_TARGET,
      "/healthcheck": API_TARGET,
      "/models": API_TARGET,
      "/docs": API_TARGET,
      "/redoc": API_TARGET,
      "/openapi.json": API_TARGET,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
