import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8282,
    proxy: {
      '/v1': {
        target: 'http://localhost:8181',
        changeOrigin: true,
      }
    }
  }
})
