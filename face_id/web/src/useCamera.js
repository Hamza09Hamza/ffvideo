import { useCallback, useEffect, useRef, useState } from 'react'

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.85

/**
 * Plain camera access + single-frame capture. No streaming, no WebSocket —
 * the REST design captures a batch of frames locally, then submits them
 * all in one request, so this only needs to turn the camera on/off and
 * grab one JPEG blob at a time on demand.
 */
export function useCamera() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState(null)

  const start = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT },
      })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setActive(true)
      return true
    } catch (err) {
      setError(`Camera error: ${err.message}`)
      setActive(false)
      return false
    }
  }, [])

  const stop = useCallback(() => {
    const stream = videoRef.current?.srcObject
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    setActive(false)
  }, [])

  const captureFrame = useCallback(() => {
    return new Promise((resolve) => {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)
      canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY)
    })
  }, [])

  useEffect(() => stop, [stop])

  return { videoRef, canvasRef, active, error, start, stop, captureFrame, CAPTURE_WIDTH, CAPTURE_HEIGHT }
}
