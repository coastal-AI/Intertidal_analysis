// Vite configuration of the MAREA map viewer (React + Tailwind CSS, built to static files for GitHub Pages).
//
// Build variables (SITE_BASE, SITE_PUBLIC_DIR, SITE_OUT_DIR, SITE_CACHE_DIR): see scripts/site-env.mjs.
// Client variables, read by the app through import.meta.env (VITE_REPO_URL, VITE_ENABLE_3D): see src/config.ts.
import { fileURLToPath } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { siteEnv } from './scripts/site-env.mjs'

const siteDir = fileURLToPath(new URL('.', import.meta.url))
const site = siteEnv(siteDir)

export default defineConfig({
  base: site.base,
  publicDir: site.publicDir,
  cacheDir: site.cacheDir,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: site.outDir,
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2022',
    // MapLibre GL JS (about 1.1 MB, in the map viewer's own chunk) is one chunk by design.
    chunkSizeWarningLimit: 1200,
  },
  server: { port: 5173 },
  preview: { port: 4173 },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    // Vitest replaces CSS imports with empty strings; the token contrast test reads index.css as text.
    css: { include: [/src[\\/]styles[\\/]index\.css/] },
  },
})
