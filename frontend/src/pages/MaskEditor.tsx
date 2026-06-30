import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiSegmentPreview, apiCreateJob } from '../api/jobs'
import type { MaskPoint } from '../api/jobs'

type Tool = 'point' | 'brush' | 'eraser'

// Convert a pointer event position to native video-pixel coordinates.
// The canvas internal size equals the video's native resolution, so this
// maps display-space coords to canvas/video-space coords consistently.
function canvasToVideoCoords(
  clientX: number,
  clientY: number,
  canvasEl: HTMLCanvasElement,
  videoNativeW: number,
  videoNativeH: number
): MaskPoint {
  const rect = canvasEl.getBoundingClientRect()
  return {
    x: ((clientX - rect.left) / rect.width) * videoNativeW,
    y: ((clientY - rect.top) / rect.height) * videoNativeH,
  }
}

// Derive canvas-internal coords (used for drawing) from the same pointer event.
function clientToCanvasCoords(
  clientX: number,
  clientY: number,
  canvasEl: HTMLCanvasElement
): { x: number; y: number } {
  const rect = canvasEl.getBoundingClientRect()
  return {
    x: ((clientX - rect.left) / rect.width) * canvasEl.width,
    y: ((clientY - rect.top) / rect.height) * canvasEl.height,
  }
}

const BRUSH_RADIUS = 12  // canvas-space radius, scales with video resolution

export default function MaskEditor() {
  const { isAuthenticated } = useAuthStore()
  const { file, s3Key, setJobId } = useUploadStore()
  const navigate = useNavigate()

  const frameCanvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const [tool, setTool] = useState<Tool>('point')
  const [frameReady, setFrameReady] = useState(false)
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 })
  const [isDrawing, setIsDrawing] = useState(false)
  const [maskPoints, setMaskPoints] = useState<MaskPoint[]>([])
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewClickPos, setPreviewClickPos] = useState<{ x: number; y: number } | null>(null)
  const [showStubBadge, setShowStubBadge] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [totalFrames, setTotalFrames] = useState(0)
  const [currentFrame, setCurrentFrame] = useState(0)

  // Auth + precondition guard
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login?redirect=/editor', { replace: true })
      return
    }
    if (!file || !s3Key) {
      navigate('/upload', { replace: true })
    }
  }, [isAuthenticated, file, s3Key, navigate])

  // Extract first frame via an off-screen video element, kept alive for scrubbing
  useEffect(() => {
    if (!file || !frameCanvasRef.current || !maskCanvasRef.current) return

    const video = document.createElement('video')
    const url = URL.createObjectURL(file)
    video.src = url
    video.muted = true
    video.preload = 'metadata'
    videoRef.current = video

    let initialized = false

    // Redraws only the video-frame canvas, never the mask overlay, so
    // scrubbing across frames doesn't disturb whatever the user has painted.
    const onSeeked = () => {
      const frameCanvas = frameCanvasRef.current
      if (!frameCanvas) return

      const ctx = frameCanvas.getContext('2d')
      if (!ctx) return

      if (!initialized) {
        const maskCanvas = maskCanvasRef.current
        frameCanvas.width = video.videoWidth
        frameCanvas.height = video.videoHeight
        if (maskCanvas) {
          maskCanvas.width = video.videoWidth
          maskCanvas.height = video.videoHeight
        }
        setVideoSize({ w: video.videoWidth, h: video.videoHeight })
        // No exact client-side fps, so estimate frame count from duration.
        const estimatedFrames = Math.max(1, Math.round(video.duration * 30))
        setTotalFrames(estimatedFrames)
        setFrameReady(true)
        initialized = true
      }

      ctx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height)
    }

    const onMeta = () => {
      video.currentTime = 0
    }

    video.addEventListener('loadedmetadata', onMeta)
    video.addEventListener('seeked', onSeeked)
    video.load()

    return () => {
      video.removeEventListener('loadedmetadata', onMeta)
      video.removeEventListener('seeked', onSeeked)
      videoRef.current = null
      URL.revokeObjectURL(url)
    }
  }, [file])

  const handleScrub = useCallback(
    (newFrame: number) => {
      setCurrentFrame(newFrame)
      const video = videoRef.current
      if (!video || totalFrames <= 0 || !Number.isFinite(video.duration)) return
      video.currentTime = (newFrame / totalFrames) * video.duration
    },
    [totalFrames]
  )

  // Draw the returned preview polygon on the mask canvas
  const drawPreviewPolygon = useCallback((points: MaskPoint[]) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas || points.length < 2) return
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y)
    }
    ctx.closePath()
    ctx.fillStyle = 'rgba(220, 38, 38, 0.35)'
    ctx.strokeStyle = 'rgba(220, 38, 38, 0.85)'
    ctx.lineWidth = 2
    ctx.fill()
    ctx.stroke()
  }, [])

  const handlePointClick = useCallback(
    async (clientX: number, clientY: number) => {
      if (!maskCanvasRef.current || !s3Key || !frameReady) return

      const rect = maskCanvasRef.current.getBoundingClientRect()
      setPreviewClickPos({ x: clientX - rect.left, y: clientY - rect.top })
      setPreviewLoading(true)

      const videoPoint = canvasToVideoCoords(
        clientX, clientY,
        maskCanvasRef.current,
        videoSize.w, videoSize.h
      )

      try {
        const response = await apiSegmentPreview(s3Key, videoPoint, currentFrame)
        drawPreviewPolygon(response.mask_points)
        if (response.stub) setShowStubBadge(true)
        // Add the clicked point to the mask data
        setMaskPoints((prev) => [...prev, videoPoint])
      } catch {
        // Non-fatal — just clear the loading dot
      } finally {
        setPreviewLoading(false)
        setPreviewClickPos(null)
      }
    },
    [s3Key, frameReady, videoSize, currentFrame, drawPreviewPolygon]
  )

  const handleBrushMove = useCallback(
    (clientX: number, clientY: number, isErase: boolean) => {
      const maskCanvas = maskCanvasRef.current
      if (!maskCanvas || !frameReady) return
      const ctx = maskCanvas.getContext('2d')
      if (!ctx) return

      const { x, y } = clientToCanvasCoords(clientX, clientY, maskCanvas)

      if (isErase) {
        const prev = ctx.globalCompositeOperation
        ctx.globalCompositeOperation = 'destination-out'
        ctx.beginPath()
        ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(0,0,0,1)'
        ctx.fill()
        ctx.globalCompositeOperation = prev
      } else {
        ctx.beginPath()
        ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(220, 38, 38, 0.5)'
        ctx.fill()
        // Track the point in video coordinates for submit
        const videoPoint = canvasToVideoCoords(clientX, clientY, maskCanvas, videoSize.w, videoSize.h)
        setMaskPoints((prev) => [...prev, videoPoint])
      }
    },
    [frameReady, videoSize]
  )

  const handleClearAll = useCallback(() => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return
    const ctx = maskCanvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    setMaskPoints([])
    setShowStubBadge(false)
  }, [])

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (tool === 'point') {
      handlePointClick(e.clientX, e.clientY)
    } else {
      setIsDrawing(true)
      handleBrushMove(e.clientX, e.clientY, tool === 'eraser')
    }
  }

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing || tool === 'point') return
    handleBrushMove(e.clientX, e.clientY, tool === 'eraser')
  }

  const handlePointerUp = () => setIsDrawing(false)

  const handleSubmit = async () => {
    if (!s3Key) return
    setSubmitError(null)
    setSubmitting(true)

    try {
      const result = await apiCreateJob(s3Key, {
        points: maskPoints,
        frame_index: currentFrame,
      })
      setJobId(result.job_id)
      navigate(`/progress/${result.job_id}`)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Failed to submit job. Please try again.'
      setSubmitError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  if (!isAuthenticated || !file || !s3Key) return null

  return (
    <div className="min-h-screen bg-[#FAFAF8] py-10 px-4">
      <div className="max-w-5xl mx-auto">
        <p className="text-sm font-semibold text-red-600 mb-4">• Mark what to remove</p>
        <h1 className="text-4xl font-black text-stone-900 tracking-tight mb-2">Mask Editor</h1>
        <p className="text-stone-500 mb-6 text-sm">
          Click, brush, or paint over the object you want erased. Then submit.
        </p>

        {showStubBadge && (
          <div className="mb-4 inline-flex items-center gap-2 px-3 py-1.5 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700 font-medium">
            <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Preview mode — full AI segmentation coming soon
          </div>
        )}

        <div className="flex gap-4 items-start">
          {/* Canvas + scrubber column */}
          <div className="flex-1 flex flex-col gap-3">
            <div className="bg-stone-900 rounded-2xl overflow-hidden relative">
              {!frameReady && (
                <div className="absolute inset-0 flex items-center justify-center text-stone-400 text-sm">
                  Loading first frame…
                </div>
              )}
              <div className="relative" style={{ lineHeight: 0 }}>
                {/* Frame canvas */}
                <canvas
                  ref={frameCanvasRef}
                  className="w-full block"
                  style={{ display: frameReady ? 'block' : 'none' }}
                />
                {/* Mask overlay canvas — exactly stacked on top */}
                <canvas
                  ref={maskCanvasRef}
                  className="w-full block absolute inset-0"
                  style={{
                    display: frameReady ? 'block' : 'none',
                    cursor: tool === 'point' ? 'crosshair' : tool === 'eraser' ? 'cell' : 'default',
                  }}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onPointerLeave={handlePointerUp}
                />
                {/* Loading dot at click position for point tool */}
                {previewLoading && previewClickPos && (
                  <div
                    className="absolute w-5 h-5 rounded-full border-2 border-red-400 border-t-transparent animate-spin"
                    style={{
                      left: previewClickPos.x - 10,
                      top: previewClickPos.y - 10,
                      pointerEvents: 'none',
                    }}
                  />
                )}
              </div>
            </div>

            {/* Frame scrubber */}
            <div className="bg-white rounded-2xl shadow-sm border border-stone-100 p-4">
              <input
                type="range"
                min={0}
                max={Math.max(totalFrames - 1, 0)}
                value={currentFrame}
                disabled={!frameReady || totalFrames <= 1}
                onChange={(e) => handleScrub(Number(e.target.value))}
                className="w-full accent-red-600 disabled:opacity-50"
              />
              <p className="text-xs text-stone-500 text-center mt-2 font-medium">
                Frame {currentFrame} / {totalFrames}
              </p>
            </div>
          </div>

          {/* Toolbar */}
          <div className="w-48 flex-shrink-0 bg-white rounded-2xl shadow-sm border border-stone-100 p-4 flex flex-col gap-3">
            <p className="text-xs font-semibold text-stone-400 uppercase tracking-wide">Tools</p>

            <button
              onClick={() => setTool('point')}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-colors ${
                tool === 'point'
                  ? 'bg-red-600 text-white'
                  : 'text-stone-600 hover:bg-stone-50'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" />
              </svg>
              Point Click
            </button>

            <button
              onClick={() => setTool('brush')}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-colors ${
                tool === 'brush'
                  ? 'bg-red-600 text-white'
                  : 'text-stone-600 hover:bg-stone-50'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
              Brush
            </button>

            <button
              onClick={() => setTool('eraser')}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-colors ${
                tool === 'eraser'
                  ? 'bg-red-600 text-white'
                  : 'text-stone-600 hover:bg-stone-50'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Eraser
            </button>

            <div className="border-t border-stone-100 my-1" />

            <button
              onClick={handleClearAll}
              className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold text-stone-500 hover:bg-stone-50 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Clear All
            </button>

            <div className="border-t border-stone-100 my-1" />

            {submitError && (
              <p className="text-xs text-red-600 bg-red-50 rounded-lg px-2 py-1.5">{submitError}</p>
            )}

            <button
              onClick={handleSubmit}
              disabled={submitting || !frameReady}
              className="w-full py-2.5 bg-red-600 text-white font-semibold rounded-xl text-sm hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? 'Submitting…' : 'Submit'}
            </button>

            <p className="text-xs text-stone-400 text-center">
              {maskPoints.length} point{maskPoints.length !== 1 ? 's' : ''} marked
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
