import { useState } from 'react'
import { useCamera } from './useCamera.js'
import { postVerify } from './api.js'

const CAPTURE_COUNT = 15
const CAPTURE_INTERVAL_MS = 100 // ~10fps over ~1.5s total

export default function VerifyPanel({ serverUrl, apiKey }) {
  const { videoRef, canvasRef, start, stop, captureFrame, error: cameraError, CAPTURE_WIDTH, CAPTURE_HEIGHT } =
    useCamera()

  const [phase, setPhase] = useState('idle') // idle | capturing | submitting | done
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const busy = phase === 'capturing' || phase === 'submitting'

  const handleStart = async () => {
    setError(null)
    setResult(null)
    const ok = await start()
    if (!ok) return

    setPhase('capturing')
    const blobs = []
    for (let i = 0; i < CAPTURE_COUNT; i++) {
      await new Promise((r) => setTimeout(r, CAPTURE_INTERVAL_MS))
      const blob = await captureFrame()
      if (blob) blobs.push(blob)
      setProgress(i + 1)
    }

    stop()
    setPhase('submitting')
    try {
      setResult(await postVerify({ serverUrl, apiKey, frameBlobs: blobs }))
    } catch (err) {
      setError(err.message)
    }
    setPhase('done')
  }

  return (
    <div className="panel">
      <div className="camera-frame" style={{ width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT }}>
        <video ref={videoRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} muted playsInline />
        <canvas ref={canvasRef} width={CAPTURE_WIDTH} height={CAPTURE_HEIGHT} style={{ display: 'none' }} />
        {phase === 'capturing' && (
          <div className="overlay">
            Hold still...
            <div className="big">{progress}/{CAPTURE_COUNT}</div>
          </div>
        )}
        {phase === 'submitting' && <div className="overlay">Verifying...</div>}
      </div>

      <button onClick={handleStart} disabled={busy}>
        Start Verification
      </button>

      {cameraError && <div className="error">{cameraError}</div>}
      {error && <div className="error">{error}</div>}
      {result && phase === 'done' && <ResultCard result={result} />}
    </div>
  )
}

function ResultCard({ result }) {
  if (result.success) {
    return (
      <div className="result success">
        <div>✓ Welcome, {result.full_name}</div>
        <div className="details">
          similarity {result.similarity.toFixed(2)} · liveness {result.avg_liveness.toFixed(2)}
        </div>
      </div>
    )
  }

  const messages = {
    insufficient_faces: `Couldn't get a clear view of your face (${result.good_frames}/${result.required} good frames). Try again with better lighting or hold still.`,
    spoof_suspected: `Liveness check failed (score ${result.avg_liveness?.toFixed(2)}). Make sure you're using a live camera, not a photo or video.`,
    not_recognized: "Face not recognized — have you been enrolled yet?",
  }

  return <div className="result failure">✗ {messages[result.reason] || result.reason}</div>
}
