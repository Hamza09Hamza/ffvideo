import { useCallback, useEffect, useRef, useState } from 'react'

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const JPEG_QUALITY = 0.8

/**
 * Shared camera-capture + ack-paced WebSocket streaming loop, used by both
 * the verification and enrollment views — they only differ in which
 * endpoint they connect to and what they do with each response.
 *
 * "Ack-paced" means: capture one frame, send it, then wait for the
 * server's response before capturing/sending the next one. A blind
 * setInterval timer lets a backlog of stale frames build up whenever the
 * server is slower than the send interval (this bit us early on — see the
 * original App.jsx history).
 */
export function useCameraWebSocket({ onMessage }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamingRef = useRef(false)

  // onMessage is a plain callback prop, not a ref value — capture the
  // latest one via a ref so the WebSocket handlers (created once, inside
  // `start`) always call the current version instead of a stale closure.
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState('idle')
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
  }, [])

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

  const start = useCallback(
    async (wsUrl) => {
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
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setStatus('streaming')
        streamingRef.current = true
        sendNextFrame(ws)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.error) {
            setError(data.error)
          } else {
            onMessageRef.current?.(data)
          }
        } catch {
          // ignore a malformed message rather than tearing down the connection
        }
        sendNextFrame(ws)
      }

      ws.onerror = () => setError('WebSocket error — is the server running and reachable?')
      ws.onclose = () => {
        streamingRef.current = false
        setConnected(false)
        setStatus('idle')
      }
    },
    [sendNextFrame],
  )

  // Stop everything if the component unmounts (e.g. switching views, hot reload)
  useEffect(() => stop, [stop])

  return { videoRef, canvasRef, connected, status, error, start, stop, CAPTURE_WIDTH, CAPTURE_HEIGHT }
}
