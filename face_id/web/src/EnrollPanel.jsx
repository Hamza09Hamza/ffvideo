import { useState } from 'react'
import { useCamera } from './useCamera.js'
import { postEnroll } from './api.js'

const POSES = [
  { key: 'center', label: 'Look straight ahead' },
  { key: 'left', label: 'Turn your head slightly left' },
  { key: 'right', label: 'Turn your head slightly right' },
  { key: 'up', label: 'Tilt your head up slightly' },
  { key: 'down', label: 'Tilt your head down slightly' },
]

const COUNTDOWN_SECONDS = 3

export default function EnrollPanel({ serverUrl, apiKey }) {
  const { videoRef, canvasRef, start, stop, captureFrame, error: cameraError, CAPTURE_WIDTH, CAPTURE_HEIGHT } =
    useCamera()

  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [phase, setPhase] = useState('idle') // idle | countdown | submitting | done
  const [poseIndex, setPoseIndex] = useState(0)
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const busy = phase === 'countdown' || phase === 'submitting'

  const handleStart = async () => {
    setError(null)
    setResult(null)
    const ok = await start()
    if (!ok) return

    setPhase('countdown')
    const blobs = {}
    for (let i = 0; i < POSES.length; i++) {
      setPoseIndex(i)
      for (let c = COUNTDOWN_SECONDS; c > 0; c--) {
        setCountdown(c)
        await new Promise((r) => setTimeout(r, 1000))
      }
      blobs[POSES[i].key] = await captureFrame()
    }

    stop()
    setPhase('submitting')
    try {
      setResult(await postEnroll({ serverUrl, apiKey, fullName, email, poseBlobs: blobs }))
    } catch (err) {
      setError(err.message)
    }
    setPhase('done')
  }

  return (
    <div className="panel">
      <div className="form-row">
        <input
          placeholder="Full name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          disabled={busy}
        />
        <input
          placeholder="Email (optional)"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={busy}
        />
      </div>

      <div className="camera-frame" style={{ width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT }}>
        <video ref={videoRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} muted playsInline />
        <canvas ref={canvasRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} style={{ display: 'none' }} />
        {phase === 'countdown' && (
          <div className="overlay">
            {POSES[poseIndex].label}
            <div className="big">{countdown}</div>
          </div>
        )}
        {phase === 'submitting' && <div className="overlay">Enrolling...</div>}
      </div>

      <button onClick={handleStart} disabled={busy || !fullName.trim()}>
        Start Enrollment
      </button>

      {cameraError && <div className="error">{cameraError}</div>}
      {error && <div className="error">{error}</div>}

      {result && phase === 'done' && (
        result.success ? (
          <div className="result success">
            ✓ Enrolled "{fullName}" as employee #{result.employee_id}
          </div>
        ) : (
          <div className="result failure">
            ✗ Retake needed:{' '}
            {Object.entries(result.failures)
              .map(([pose, reason]) => `${pose} (${reason})`)
              .join(', ')}
          </div>
        )
      )}
    </div>
  )
}
