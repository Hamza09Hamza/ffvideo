import { useCallback, useState } from 'react'
import { useCameraWebSocket } from './useCameraWebSocket.js'

export default function VerifyPanel() {
  const [serverUrl, setServerUrl] = useState('wss://ibsai.tailff4c89.ts.net:8000/ws/verify')
  const [faces, setFaces] = useState([])
  const [timing, setTiming] = useState(null)

  const onMessage = useCallback((data) => {
    setFaces(data.faces || [])
    setTiming(data.timing_ms || null)
  }, [])

  const { videoRef, canvasRef, connected, status, error, start, stop, CAPTURE_WIDTH, CAPTURE_HEIGHT } =
    useCameraWebSocket({ onMessage })

  return (
    <div>
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
        <button onClick={() => start(serverUrl)} disabled={connected} style={{ marginLeft: 8 }}>
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
  const { box, warming_up: warmingUp, antispoof, identity, frames_seen: framesSeen, window_size: windowSize } = face

  let color = '#facc15' // yellow: still collecting frames for the sliding window
  let label = `warming up (${framesSeen}/${windowSize})`

  if (!warmingUp) {
    if (!antispoof.is_real) {
      color = '#ef4444' // red
      label = 'SPOOF'
    } else if (identity) {
      color = '#22c55e' // green
      label = `${identity.full_name} (${identity.similarity.toFixed(2)})`
    } else {
      color = '#22c55e' // green
      label = 'REAL — unrecognized'
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
