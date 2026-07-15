import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiSegmentPreview, apiCreateJob } from '../api/jobs'
import type { MaskPoint, SAM2Point } from '../api/jobs'

type Tool = 'point' | 'brush' | 'eraser' | 'sam2'

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

// Recolor a grayscale (0/255) mask PNG into a semi-transparent purple overlay,
// scaled to the target canvas size. Runs synchronously once the image has loaded.
function tintMaskPurple(img: HTMLImageElement, targetW: number, targetH: number): HTMLCanvasElement {
  const off = document.createElement('canvas')
  off.width = img.naturalWidth
  off.height = img.naturalHeight
  const octx = off.getContext('2d')!
  octx.drawImage(img, 0, 0)
  const imgData = octx.getImageData(0, 0, off.width, off.height)
  const data = imgData.data
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 127) {
      data[i] = 124
      data[i + 1] = 58
      data[i + 2] = 237
      data[i + 3] = Math.round(0.4 * 255)
    } else {
      data[i + 3] = 0
    }
  }
  octx.putImageData(imgData, 0, 0)

  if (off.width === targetW && off.height === targetH) return off
  const scaled = document.createElement('canvas')
  scaled.width = targetW
  scaled.height = targetH
  scaled.getContext('2d')!.drawImage(off, 0, 0, targetW, targetH)
  return scaled
}

const BRUSH_RADIUS = 12  // canvas-space radius, scales with video resolution
const SAM2_POINT_RADIUS = 5  // canvas-space radius for positive/negative point dots

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
  const [submitError, setSubmitError] = useState<string | null>(null)

  // SAM2 interactive segmentation state
  const [sam2Points, setSam2Points] = useState<SAM2Point[]>([])
  const [sam2Mode, setSam2Mode] = useState<1 | -1>(1)  // label used by a plain left-click
  const [sam2Loading, setSam2Loading] = useState(false)
  const [sam2Score, setSam2Score] = useState<number | null>(null)
  const [sam2Error, setSam2Error] = useState<string | null>(null)
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

  // Draw a SAM2 mask (purple overlay) plus positive/negative point dots on the mask canvas.
  const drawMaskOverlay = useCallback(
    (maskBase64: string | null, points: SAM2Point[]) => {
      const maskCanvas = maskCanvasRef.current
      if (!maskCanvas) return
      const ctx = maskCanvas.getContext('2d')
      if (!ctx) return

      const drawPoints = () => {
        for (const p of points) {
          const cx = (p.x / videoSize.w) * maskCanvas.width
          const cy = (p.y / videoSize.h) * maskCanvas.height
          ctx.beginPath()
          ctx.arc(cx, cy, SAM2_POINT_RADIUS, 0, Math.PI * 2)
          ctx.fillStyle = p.label === 1 ? '#22C55E' : '#EF4444'
          ctx.fill()
          ctx.strokeStyle = 'rgba(0,0,0,0.6)'
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }

      if (!maskBase64) {
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
        drawPoints()
        return
      }

      const img = new Image()
      img.onload = () => {
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
        const tinted = tintMaskPurple(img, maskCanvas.width, maskCanvas.height)
        ctx.drawImage(tinted, 0, 0)
        drawPoints()
      }
      img.src = `data:image/png;base64,${maskBase64}`
    },
    [videoSize]
  )

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
      const promptPoint: SAM2Point = { ...videoPoint, label: 1 }

      try {
        const response = await apiSegmentPreview(s3Key, [promptPoint], currentFrame)
        drawMaskOverlay(response.mask_base64, [promptPoint])
        // Add the clicked point to the mask data
        setMaskPoints((prev) => [...prev, videoPoint])
      } catch {
        // Non-fatal — just clear the loading dot
      } finally {
        setPreviewLoading(false)
        setPreviewClickPos(null)
      }
    },
    [s3Key, frameReady, videoSize, currentFrame, drawMaskOverlay]
  )

  const handleSam2Click = useCallback(
    async (clientX: number, clientY: number, explicitLabel?: 1 | -1) => {
      if (!maskCanvasRef.current || !s3Key || !frameReady) return

      const videoPoint = canvasToVideoCoords(
        clientX, clientY,
        maskCanvasRef.current,
        videoSize.w, videoSize.h
      )
      const label = explicitLabel ?? sam2Mode
      const nextPoints = [...sam2Points, { ...videoPoint, label }]
      setSam2Points(nextPoints)
      setSam2Loading(true)
      setSam2Error(null)

      try {
        const response = await apiSegmentPreview(s3Key, nextPoints, currentFrame)
        setSam2Score(response.score)
        drawMaskOverlay(response.mask_base64, nextPoints)
      } catch {
        setSam2Error('Segmentation failed — try again or switch to the brush tool.')
        drawMaskOverlay(null, nextPoints)
      } finally {
        setSam2Loading(false)
      }
    },
    [s3Key, frameReady, videoSize, currentFrame, sam2Points, sam2Mode, drawMaskOverlay]
  )

  const handleClearSam2Points = useCallback(() => {
    setSam2Points([])
    setSam2Score(null)
    setSam2Error(null)
    const maskCanvas = maskCanvasRef.current
    if (maskCanvas) {
      maskCanvas.getContext('2d')?.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    }
  }, [])

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
        ctx.fillStyle = 'rgba(124, 58, 237, 0.5)'
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
    setSam2Points([])
    setSam2Score(null)
    setSam2Error(null)
  }, [])

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (tool === 'point') {
      handlePointClick(e.clientX, e.clientY)
    } else if (tool === 'sam2') {
      handleSam2Click(e.clientX, e.clientY, e.button === 2 ? -1 : undefined)
    } else {
      setIsDrawing(true)
      handleBrushMove(e.clientX, e.clientY, tool === 'eraser')
    }
  }

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing || tool === 'point' || tool === 'sam2') return
    handleBrushMove(e.clientX, e.clientY, tool === 'eraser')
  }

  const handlePointerUp = () => setIsDrawing(false)

  const handleSubmit = async () => {
    if (!s3Key) return
    setSubmitError(null)
    setSubmitting(true)

    try {
      // If a SAM2 mask was generated, its positive points drive real
      // re-segmentation on the backend job pipeline; any brush edits made
      // afterwards (in maskPoints) refine that mask further.
      const sam2PositivePoints = sam2Points
        .filter((p) => p.label === 1)
        .map(({ x, y }) => ({ x, y }))
      const submittedPoints =
        sam2PositivePoints.length > 0 ? [...sam2PositivePoints, ...maskPoints] : maskPoints

      const result = await apiCreateJob(s3Key, {
        points: submittedPoints,
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

  const toolButtonClass = (active: boolean) =>
    `w-10 h-10 flex items-center justify-center rounded-xl transition-colors duration-200 ${
      active ? 'bg-[#7C3AED] text-white' : 'text-[#A0A0B0] hover:text-white hover:bg-white/5'
    }`

  return (
    <div className="h-screen flex flex-col bg-[#0A0A0F] overflow-hidden">
      {/* Top bar */}
      <div className="h-12 shrink-0 flex items-center justify-between px-4 border-b border-white/10 backdrop-blur-xl bg-white/[0.02]">
        <span className="text-xs text-[#A0A0B0] font-mono truncate max-w-[200px]">{file.name}</span>
        <span className="text-sm font-black bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent">
          VisionErase
        </span>
        <svg className="w-5 h-5 text-[#A0A0B0]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>

      <div className="flex-1 flex min-h-0">
        {/* Left tool panel */}
        <div className="w-[60px] shrink-0 flex flex-col items-center gap-2 py-4 border-r border-white/10 backdrop-blur-xl bg-white/[0.02]">
          <button onClick={() => setTool('point')} className={toolButtonClass(tool === 'point')} title="Point Click">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" />
            </svg>
          </button>
          <button onClick={() => setTool('brush')} className={toolButtonClass(tool === 'brush')} title="Brush">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </button>
          <button onClick={() => setTool('eraser')} className={toolButtonClass(tool === 'eraser')} title="Eraser">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
          <button onClick={() => setTool('sam2')} className={toolButtonClass(tool === 'sam2')} title="SAM2 (click to segment)">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
            </svg>
          </button>
          <div className="w-6 h-px bg-white/10 my-1" />
          <button onClick={handleClearAll} className={toolButtonClass(false)} title="Clear All">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16M9 7V4a1 1 0 011-1h4a1 1 0 011 1v3" />
            </svg>
          </button>
        </div>

        {/* Center canvas area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 flex items-center justify-center p-6 relative overflow-auto">
            {tool === 'sam2' && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-3 py-1.5 backdrop-blur-xl bg-white/[0.04] border border-white/10 rounded-full text-xs">
                <button
                  onClick={() => setSam2Mode((m) => (m === 1 ? -1 : 1))}
                  className={`px-2 py-0.5 rounded-full font-medium transition-colors ${
                    sam2Mode === 1
                      ? 'bg-emerald-500/15 text-emerald-400'
                      : 'bg-red-500/15 text-red-400'
                  }`}
                  title="Left-click uses this label. Right-click is always negative."
                >
                  {sam2Mode === 1 ? '+ Positive' : '− Negative'}
                </button>
                <button
                  onClick={handleClearSam2Points}
                  className="px-2 py-0.5 rounded-full text-[#A0A0B0] hover:text-white hover:bg-white/5 font-medium"
                >
                  Clear Points
                </button>
                {sam2Loading && (
                  <span className="inline-flex items-center gap-1.5 text-[#A78BFA] font-medium">
                    <span className="w-3 h-3 rounded-full border-2 border-[#A78BFA] border-t-transparent animate-spin" />
                    Segmenting…
                  </span>
                )}
                {!sam2Loading && sam2Score !== null && (
                  <span
                    className={`px-2 py-0.5 rounded-full font-mono font-medium ${
                      sam2Score >= 0.7
                        ? 'bg-emerald-500/15 text-emerald-400'
                        : 'bg-amber-500/15 text-amber-400'
                    }`}
                  >
                    Confidence: {Math.round(sam2Score * 100)}%
                  </span>
                )}
                {sam2Error && <span className="text-[#EF4444] font-medium">{sam2Error}</span>}
              </div>
            )}

            <div
              className="relative rounded-lg overflow-hidden"
              style={{ boxShadow: '0 0 40px rgba(124,58,237,0.2)', lineHeight: 0 }}
            >
              {!frameReady && (
                <div className="w-[480px] h-[270px] flex items-center justify-center text-[#A0A0B0] text-sm bg-[#12121A]">
                  Loading first frame…
                </div>
              )}
              {/* Frame canvas */}
              <canvas
                ref={frameCanvasRef}
                className="block max-w-full max-h-[calc(100vh-220px)]"
                style={{ display: frameReady ? 'block' : 'none' }}
              />
              {/* Mask overlay canvas — exactly stacked on top */}
              <canvas
                ref={maskCanvasRef}
                className="block absolute inset-0 max-w-full max-h-[calc(100vh-220px)] w-full h-full"
                style={{
                  display: frameReady ? 'block' : 'none',
                  cursor:
                    tool === 'point' || tool === 'sam2'
                      ? 'crosshair'
                      : tool === 'eraser'
                        ? 'cell'
                        : 'default',
                }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                onContextMenu={(e) => {
                  if (tool === 'sam2') e.preventDefault()
                }}
              />
              {/* Loading dot at click position for point tool */}
              {previewLoading && previewClickPos && (
                <div
                  className="absolute w-5 h-5 rounded-full border-2 border-[#A78BFA] border-t-transparent animate-spin"
                  style={{
                    left: previewClickPos.x - 10,
                    top: previewClickPos.y - 10,
                    pointerEvents: 'none',
                  }}
                />
              )}
            </div>
          </div>

          {/* Bottom panel: scrubber + controls */}
          <div className="shrink-0 border-t border-white/10 backdrop-blur-xl bg-white/[0.02] px-6 py-3 flex items-center gap-4">
            <input
              type="range"
              min={0}
              max={Math.max(totalFrames - 1, 0)}
              value={currentFrame}
              disabled={!frameReady || totalFrames <= 1}
              onChange={(e) => handleScrub(Number(e.target.value))}
              className="flex-1 accent-purple disabled:opacity-50"
              style={{ accentColor: '#7C3AED' }}
            />
            <span className="text-xs font-mono text-[#A0A0B0] shrink-0">
              Frame {currentFrame} / {totalFrames}
            </span>
            <span className="text-xs text-[#A0A0B0] shrink-0">
              {maskPoints.length} point{maskPoints.length !== 1 ? 's' : ''} marked
              {sam2Points.length > 0 && ` · ${sam2Points.length} SAM2 point${sam2Points.length !== 1 ? 's' : ''}`}
            </span>
            {submitError && (
              <span className="text-xs text-[#EF4444] shrink-0">{submitError}</span>
            )}
            <button
              onClick={handleSubmit}
              disabled={submitting || !frameReady}
              className="shrink-0 px-5 py-2 bg-gradient-to-r from-[#7C3AED] to-[#2563EB] text-white font-semibold rounded-xl text-sm shadow-[0_0_16px_rgba(124,58,237,0.35)] hover:shadow-[0_0_24px_rgba(124,58,237,0.55)] transition-shadow duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
            >
              {submitting ? 'Submitting…' : 'Start Removal →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
