import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// getUserMedia (camera access) only works in a "secure context": HTTPS, or
// literally `localhost`. Since this is accessed over Tailscale at a real
// hostname (not localhost), it needs real TLS — hence the tailscale-issued
// cert/key living in ../../certs (see: tailscale cert).
const certDir = path.resolve(__dirname, '../../certs')

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // listen on all interfaces so you can reach it over the tailnet too
    port: 5173,
    https: {
      key: fs.readFileSync(path.join(certDir, 'tailscale.key')),
      cert: fs.readFileSync(path.join(certDir, 'tailscale.crt')),
    },
  },
})
