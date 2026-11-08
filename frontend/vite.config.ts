import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // MapLibre 6 loads its worker relative to its own module; pre-bundling moves
  // the module away from the worker file and the map never renders.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  worker: { format: 'es' },
})
