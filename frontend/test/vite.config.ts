import { defineConfig } from "vite";
import baseConfig from "../vite.config";

// Isolated test server: the user-facing development server keeps the real API.
export default defineConfig({
  ...baseConfig,
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:18000" },
  },
});
