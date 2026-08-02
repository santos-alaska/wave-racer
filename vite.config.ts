import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      // The screenshot harness writes PNGs into the repo. Without this, vite
      // sees the write, triggers a full reload, and destroys the in-flight
      // page.evaluate — which silently truncated capture runs mid-shot-list.
      ignored: ['**/shots/**', '**/.scratch*/**'],
    },
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Keep three in its own chunk so the game code stays cache-friendly.
        manualChunks: { three: ['three'] },
      },
    },
  },
});
