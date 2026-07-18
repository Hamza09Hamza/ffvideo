import { useCallback, useEffect, useState } from 'react'
import { useCameraWebSocket } from './useCameraWebSocket.js'

export default function EnrollPanel() {
  const [serverHost, setServerHost] = useState('wss://ibsai.tailff4c89.ts.net:8000')
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [progress, setProgress] = useState(null) // last status dict from the server
  const [enrolledId, setEnrolledId] = useState(null)

  const onMessage = useCallback((data) => {
    setProgress(data)
    if (data.status === 'enrolled') setEnrolledId(data.employee_id)
  }, [])

  const { videoRef, canvasRef, connected, status, error, start, stop, CAPTURE_WIDTH, CAPTURE_HEIGHT } =
    useCameraWebSocket({ onMessage })

  // Release the camera + close the socket once the server confirms enrollment.
  useEffect(() => {
    if (enrolledId !== null) stop()
  }, [enrolledId, stop])

  const handleStart = () => {
    setEnrolledId(null)
    setProgress(null)
    const params = new URLSearchParams({ full_name: fullName })
    if (email) params.set('email', email)
    start(`${serverHost}/ws/enroll?${params.toString()}`)
  }

  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <label>
          Server:{' '}
          <input
            value={serverHost}
            onChange={(e) => setServerHost(e.target.value)}
            disabled={connected}
            style={{ width: 280 }}
          />
        </label>
      </div>

      <div style={{ marginBottom: 12 }}>
        <label>
          Full name:{' '}
          <input value={fullName} onChange={(e) => setFullName(e.target.value)} disabled={connected} />
        </label>
        <label style={{ marginLeft: 12 }}>
          Email (optional):{' '}
          <input value={email} onChange={(e) => setEmail(e.target.value)} disabled={connected} />
        </label>
        <button onClick={handleStart} disabled={connected || !fullName.trim()} style={{ marginLeft: 12 }}>
          Start Enrollment
        </button>
        <button onClick={stop} disabled={!connected} style={{ marginLeft: 8 }}>
          Stop
        </button>
        <span style={{ marginLeft: 12 }}>status: {status}</span>
      </div>

      {error && <div style={{ color: 'red', marginBottom: 12 }}>{error}</div>}

      {enrolledId !== null && (
        <div style={{ color: '#22c55e', marginBottom: 12, fontWeight: 'bold' }}>
          ✓ Enrolled "{fullName}" as employee #{enrolledId}
        </div>
      )}

      {progress && enrolledId === null && (
        <div style={{ marginBottom: 12, fontFamily: 'monospace', fontSize: 14 }}>
          {describeProgress(progress)}
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
      </div>

      {/* Offscreen canvas: grabs a JPEG snapshot of the video each send tick */}
      <canvas ref={canvasRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} style={{ display: 'none' }} />
    </div>
  )
}

function describeProgress({ status, captured, required }) {
  switch (status) {
    case 'no_face':
      return `No face detected — captured ${captured}/${required}`
    case 'multiple_faces':
      return `Multiple faces in frame, only one allowed — captured ${captured}/${required}`
    case 'spoof_suspected':
      return `Liveness check failed on that frame — captured ${captured}/${required}`
    case 'capturing':
      return `Capturing... ${captured}/${required} good frames`
    default:
      return status
  }
}
