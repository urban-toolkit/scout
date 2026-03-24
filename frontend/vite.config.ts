import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    base: mode === "production" ? "/scout/" : "/",
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/api": {
          target: env.VITE_BACKEND_ORIGIN || "http://localhost:5000",
          changeOrigin: true,
        },
        "/generated": {
          target: env.VITE_BACKEND_ORIGIN || "http://localhost:5000",
          changeOrigin: true,
        },
      },
    },
  };
});
