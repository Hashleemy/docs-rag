import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the React dev server proxies /api to the FastAPI backend.
// In production FastAPI serves the built files from web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
