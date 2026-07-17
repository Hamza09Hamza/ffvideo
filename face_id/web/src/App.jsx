import { useCallback, useEffect, useRef, useState } from 'react'

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.8

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null) // offscreen: only used to JPEG-encode the current video frame
  const wsRef = useRef(null)
  const streamingRef = useRef(false) // guards the send-loop so it stops cleanly, not via a timer

  const [serverUrl, setServerUrl] = useState('wss://ibsai.tailff4c89.ts.net:8000/ws/verify')
  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState('idle')
  const [faces, setFaces] = useState([])
  const [timing, setTiming] = useState(null)
  const [error, setError] = useState(null)

  const stop = useCallback(() => {
    streamingRef.current = false
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    const stream = videoRef.current?.srcObject
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    setConnected(false)
    setStatus('idle')
    setFaces([])
  }, [])

  // Capture + send exactly one frame. Called once to kick off the loop, then
  // again from onmessage each time the server finishes the previous frame —
  // so we never send faster than the server can actually keep up with, and
  // never let a backlog of stale frames build up in the socket's send buffer.
  const sendNextFrame = useCallback((ws) => {
    if (!streamingRef.current || ws.readyState !== WebSocket.OPEN) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    canvas.toBlob(
      (blob) => {
        if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob)
      },
      'image/jpeg',
      JPEG_QUALITY,
    )
  }, [])

  const start = useCallback(async () => {
    setError(null)
    setStatus('requesting camera...')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT },
      })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    } catch (err) {
      setError(`Camera error: ${err.message}`)
      setStatus('idle')
      return
    }

    setStatus('connecting...')
    const ws = new WebSocket(serverUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setStatus('streaming')
      streamingRef.current = true
      sendNextFrame(ws) // kick off the loop; onmessage keeps it going
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
          setError(data.error)
        } else {
          setFaces(data.faces || [])
          setTiming(data.timing_ms || null)
        }
      } catch {
        // ignore a malformed message rather than tearing down the connection
      }
      sendNextFrame(ws) // previous frame is fully processed — safe to send the next one
    }

    ws.onerror = () => setError('WebSocket error — is the server running and reachable?')
    ws.onclose = () => {
      streamingRef.current = false
      setConnected(false)
      setStatus('idle')
    }
  }, [serverUrl, sendNextFrame])

  // Stop everything if the component unmounts (e.g. hot reload)
  useEffect(() => stop, [stop])

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 20 }}>
      <h1>Face ID — Verification Test</h1>

      <div style={{ marginBottom: 12 }}>
        <label>
          Server:{' '}
          <input
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            disabled={connected}
            style={{ width: 320 }}
          />
        </label>
        <button onClick={start} disabled={connected} style={{ marginLeft: 8 }}>
          Start
        </button>
        <button onClick={stop} disabled={!connected} style={{ marginLeft: 8 }}>
          Stop
        </button>
        <span style={{ marginLeft: 12 }}>status: {status}</span>
      </div>

      {error && <div style={{ color: 'red', marginBottom: 12 }}>{error}</div>}

      {timing && (
        <div style={{ marginBottom: 12, fontFamily: 'monospace', fontSize: 13 }}>
          per-frame: detect {timing.detect}ms | antispoof {timing.antispoof}ms | total {timing.total}ms
        </div>
      )}

      <div style={{ position: 'relative', width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT }}>
        <video
          ref={videoRef}
          width={CAPTURE_WIDTH}
          height={CAPTURE_HEIGHT}
          muted
          playsInline
          style={{ position: 'absolute', top: 0, left: 0, background: '#000' }}
        />
        {faces.map((face) => (
          <FaceBox key={face.face_id} face={face} />
        ))}
      </div>

      {/* Offscreen canvas: grabs a JPEG snapshot of the video each send tick */}
      <canvas ref={canvasRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} style={{ display: 'none' }} />
    </div>
  )
}

function FaceBox({ face }) {
  const { box, warming_up: warmingUp, antispoof, frames_seen: framesSeen, window_size: windowSize } = face

  let color = '#facc15' // yellow: still collecting frames for the sliding window
  let label = `warming up (${framesSeen}/${windowSize})`

  if (!warmingUp) {
    if (antispoof.is_real) {
      color = '#22c55e' // green
      label = 'VERIFIED REAL'
    } else {
      color = '#ef4444' // red
      label = 'SPOOF'
    }
  }

  const boxStyle = {
    position: 'absolute',
    left: box.x1,
    top: box.y1,
    width: box.x2 - box.x1,
    height: box.y2 - box.y1,
    border: `3px solid ${color}`,
    boxSizing: 'border-box',
  }

  return (
    <div style={boxStyle}>
      <div
        style={{
          position: 'absolute',
          top: -44,
          left: 0,
          background: color,
          color: '#000',
          fontSize: 12,
          padding: '2px 4px',
          whiteSpace: 'nowrap',
        }}
      >
        <div>{label}</div>
        {!warmingUp && <div>live: {antispoof.avg_score.toFixed(2)}</div>}
      </div>
    </div>
  )
}
