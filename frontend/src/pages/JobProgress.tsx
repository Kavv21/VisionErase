import { useEffect, useRef, useState, type ReactElement } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuthStore } from '../store/authStore'
import { apiGetJob } from '../api/jobs'
import ProgressRing from '../components/ProgressRing'
import StageCard, { type StageStatus } from '../components/StageCard'
import GlassCard from '../components/GlassCard'

// Message shapes the /ws/{job_id} endpoint actually emits (see api/routers/websocket.py)
// chunk/total are present once the pipeline splits a long video into chunks
// (see workers/segmentation/chunk_tasks.py, workers/inpainting/chunk_tasks.py).
// chunk is 1-based.
type WsMsg =
  | { type: 'progress'; stage: string; pct: number; chunk?: number; total?: number }
  | { type: 'complete'; result: { result_s3_key: string | null } }
  | { type: 'error'; message: string }
  // pub/sub forwarded messages from workers lack a `type` field
  | { stage: string; pct: number; chunk?: number; total?: number }

const STAGE_LABELS: Record<string, string> = {
  pending: 'Waiting to start',
  segmenting: 'SAM2 Object Mask Generated',
  tracking: 'XMem++ Tracking Mask Across Frames',
  oiv: 'OIV Embedding — Verifying Object Identity',
  inpainting: 'ProPainter — Filling Removed Region',
  stitching: 'Stitching Final Video',
  boundary_fusion: 'BoundaryFusion — Cleaning Edges',
  quality: 'Quality Check',
  quality_check: 'Quality Check',
  completed: 'Processing Complete',
  failed: 'Failed',
}

// Map a worker progress event to an overall 0-100 job percentage.
// Stage budget: segmenting 0-5, tracking 5-45, inpainting 45-80,
// stitching 83, boundary_fusion 90, quality 96, completed 100.
// Must stay in sync with api/core/progress.py (compute_overall_pct).
const computeOverallPct = (msg: { stage?: string; pct?: number; chunk?: number; total?: number }): number => {
  const stage = msg.stage || ''
  // chunk is 1-based; messages without chunk info span the whole stage
  const chunk = msg.chunk || 1
  const total = msg.total || 1
  const pct = msg.pct || 0

  if (stage === 'segmenting' || stage === 'segmentation') return 5

  if (stage === 'tracking') {
    // Each chunk is 1/total of the 40% tracking phase
    const chunkFraction = (chunk - 1 + pct / 100) / total
    return Math.min(45, Math.max(5, Math.round(5 + chunkFraction * 40))) // 5-45%
  }

  if (stage === 'inpainting') {
    const chunkFraction = (chunk - 1 + pct / 100) / total
    return Math.min(80, Math.max(45, Math.round(45 + chunkFraction * 35))) // 45-80%
  }

  if (stage === 'stitching') return 83
  if (stage === 'boundary_fusion') return 90
  if (stage === 'quality' || stage === 'quality_check') return 96
  if (stage === 'completed') return 100

  // Fallback: use pct directly if available
  return pct || 0
}

// Backend stage progression, used only to derive which display stage is active/complete.
const BACKEND_ORDER = ['pending', 'segmenting', 'tracking', 'inpainting', 'stitching', 'boundary_fusion', 'quality_check', 'completed']

// stageKey marks the card that shows "chunk X of Y" while that stage is active.
const PIPELINE_STAGES: { title: string; backendIdx: number; stageKey?: string; icon: ReactElement }[] = [
  {
    title: 'Uploading Video',
    backendIdx: 0,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
      </svg>
    ),
  },
  {
    title: 'SAM2 Object Mask Generated',
    backendIdx: 1,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" />
      </svg>
    ),
  },
  {
    title: 'XMem++ Tracking Mask Across Frames',
    backendIdx: 2,
    stageKey: 'tracking',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    title: 'OIV Embedding — Verifying Object Identity',
    backendIdx: 2,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  {
    title: 'ProPainter — Filling Removed Region',
    backendIdx: 3,
    stageKey: 'inpainting',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
      </svg>
    ),
  },
  {
    title: 'Stitching Final Video',
    backendIdx: 4,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    title: 'BoundaryFusion — Cleaning Edges',
    backendIdx: 5,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    ),
  },
  {
    title: 'Quality Check',
    backendIdx: 6,
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
]

function stageStatus(uiIdx: number, backendIdx: number, currentIdx: number, wsError: boolean): StageStatus {
  if (uiIdx === 0) return 'complete'
  if (currentIdx > backendIdx) return 'complete'
  if (currentIdx === backendIdx) return wsError ? 'failed' : 'active'
  return 'pending'
}

type ConnectionState = 'connecting' | 'open' | 'reconnecting' | 'failed'

interface LogLine {
  time: string
  message: string
}

export default function JobProgress() {
  const { jobId } = useParams<{ jobId: string }>()
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  const [connState, setConnState] = useState<ConnectionState>('connecting')
  const [stage, setStage] = useState('pending')
  const [overallPct, setOverallPct] = useState(0)
  const [chunkInfo, setChunkInfo] = useState<{ chunk: number; total: number } | null>(null)
  const [isDone, setIsDone] = useState(false)
  const [resultKey, setResultKey] = useState<string | null>(null)
  const [wsError, setWsError] = useState<string | null>(null)

  const reconnectAttemptedRef = useRef(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!isAuthenticated) {
      navigate(`/login?redirect=/progress/${jobId}`, { replace: true })
    }
  }, [isAuthenticated, jobId, navigate])

  useEffect(() => {
    if (!jobId || !isAuthenticated) return

    const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
    const wsBase = apiBase.replace(/^http/, 'ws')

    // Overall progress is monotone: parallel chunk workers can emit events out
    // of order (e.g. tracking chunk 3 after inpainting chunk 1 started), so
    // never let the ring move backwards.
    const applyProgress = (msg: { stage?: string; pct?: number; chunk?: number; total?: number }) => {
      setStage(msg.stage ?? 'pending')
      setChunkInfo(
        typeof msg.chunk === 'number' && typeof msg.total === 'number'
          ? { chunk: msg.chunk, total: msg.total }
          : null
      )
      setOverallPct((prev) => Math.max(prev, computeOverallPct(msg)))
    }

    function connect(isReconnect: boolean) {
      if (isReconnect) {
        setConnState('reconnecting')
      } else {
        setConnState('connecting')
      }

      const ws = new WebSocket(`${wsBase}/ws/${jobId}`)
      wsRef.current = ws

      ws.addEventListener('open', () => {
        setConnState('open')
      })

      ws.addEventListener('message', (event) => {
        let msg: WsMsg
        try {
          msg = JSON.parse(event.data as string) as WsMsg
        } catch {
          return
        }

        if ('type' in msg) {
          if (msg.type === 'progress') {
            applyProgress(msg)
          } else if (msg.type === 'complete') {
            setStage('completed')
            setOverallPct(100)
            setChunkInfo(null)
            setResultKey(msg.result?.result_s3_key ?? null)
            setIsDone(true)
            ws.close()
          } else if (msg.type === 'error') {
            setWsError(msg.message ?? 'An error occurred')
            setIsDone(true)
            ws.close()
          }
        } else if ('stage' in msg) {
          // pub/sub forwarded worker event
          applyProgress(msg)
        }
      })

      ws.addEventListener('close', () => {
        if (isDone) return
        // Attempt one automatic reconnect
        if (!isReconnect && !reconnectAttemptedRef.current) {
          reconnectAttemptedRef.current = true
          setTimeout(() => connect(true), 2000)
        } else {
          setConnState('failed')
        }
      })

      ws.addEventListener('error', () => {
        ws.close()
      })
    }

    connect(false)

    return () => {
      wsRef.current?.close()
    }
  }, [jobId, isAuthenticated]) // eslint-disable-line react-hooks/exhaustive-deps

  // Polling fallback: the WebSocket completion event can be missed if the
  // browser was refreshed or the connection dropped, so also poll status
  // directly until the job finishes.
  useEffect(() => {
    if (!jobId || !isAuthenticated || isDone) return

    const interval = setInterval(async () => {
      try {
        const data = await apiGetJob(jobId)
        if (data.status === 'completed') {
          setStage('completed')
          setOverallPct(100)
          setResultKey(data.result_s3_key ?? null)
          setIsDone(true)
        } else if (data.status === 'failed') {
          setWsError(data.error ?? 'An error occurred')
          setIsDone(true)
        } else {
          // REST progress_pct is already the overall percentage
          // (computed server-side from the latest worker progress event).
          setOverallPct((prev) => Math.max(prev, data.progress_pct))
          setStage((prev) => {
            // REST status has no chunk granularity — only drop the chunk
            // label when the stage actually moved on.
            if (prev !== data.status) setChunkInfo(null)
            return data.status
          })
        }
      } catch {
        // transient polling error — the next tick will retry
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [jobId, isAuthenticated, isDone])

  // Once the job is confirmed complete (via either WebSocket or polling),
  // move on to the result page automatically.
  useEffect(() => {
    if (isDone && !wsError && jobId) {
      navigate(`/result/${jobId}`)
    }
  }, [isDone, wsError, jobId, navigate])

  // Chunk-aware label, e.g. "XMem++ Tracking Mask Across Frames — chunk 2 of 5"
  // once the pipeline has split the video into chunks (chunk is 1-based).
  const stageLabel = chunkInfo
    ? `${STAGE_LABELS[stage] ?? stage} — chunk ${chunkInfo.chunk} of ${chunkInfo.total}`
    : STAGE_LABELS[stage] ?? stage

  // Purely visual: append a timestamped line to the processing log whenever
  // the displayed stage (or chunk within a stage) changes. Does not affect
  // WS/polling behavior.
  const [log, setLog] = useState<LogLine[]>([])
  const lastLoggedKeyRef = useRef<string | null>(null)
  useEffect(() => {
    const key = chunkInfo ? `${stage}:${chunkInfo.chunk}` : stage
    if (lastLoggedKeyRef.current === key) return
    lastLoggedKeyRef.current = key
    const time = new Date().toLocaleTimeString([], { hour12: false })
    setLog((prev) => [...prev, { time, message: stageLabel }])
  }, [stage, chunkInfo, stageLabel])
  const logEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  if (!isAuthenticated) return null

  const currentIdx = stage === 'completed' ? BACKEND_ORDER.length - 1 : BACKEND_ORDER.indexOf(stage)

  return (
    <div className="min-h-screen bg-[#0A0A0F] py-16 px-4">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl md:text-5xl font-black tracking-tight mb-2 bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent">
          Processing Your Video
        </h1>
        <p className="text-[#A0A0B0] text-xs font-mono mb-10 select-all">{jobId}</p>

        {/* Waiting to start */}
        {stage === 'pending' && !wsError && connState !== 'failed' ? (
          <GlassCard className="p-10 flex flex-col items-center text-center mb-8">
            <motion.div
              animate={{ scale: [1, 1.15, 1], opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              className="w-16 h-16 rounded-full bg-[#7C3AED] shadow-[0_0_40px_rgba(124,58,237,0.6)] mb-5"
            />
            <p className="text-[#F8F8FF] font-semibold mb-1">Your job is queued...</p>
            <p className="text-xs text-[#A0A0B0]">
              Progress will update automatically when a worker picks it up.
            </p>
          </GlassCard>
        ) : (
          <div className="flex flex-col items-center mb-8">
            <ProgressRing percent={overallPct} />
            <p className="mt-4 text-sm text-[#A0A0B0]">{stageLabel}</p>
          </div>
        )}

        {(connState === 'connecting' || connState === 'reconnecting') && (
          <div className="flex items-center justify-center gap-2 text-[#A0A0B0] text-sm mb-6">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            {connState === 'connecting' ? 'Connecting…' : 'Reconnecting…'}
          </div>
        )}
        {connState === 'failed' && !isDone && (
          <GlassCard className="mb-6 px-4 py-3 text-sm text-amber-400 border-amber-500/30">
            <p className="font-semibold mb-1">Connection lost</p>
            <p className="mb-3 text-xs text-[#A0A0B0]">
              The server is not sending updates right now. This is expected while workers are
              starting up. The job is still queued and will process when a worker is available.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="text-xs font-semibold underline"
            >
              Refresh page
            </button>
          </GlassCard>
        )}

        {wsError && (
          <GlassCard className="mb-6 px-4 py-3 text-sm text-[#EF4444] border-[#EF4444]/30">
            <p className="font-semibold mb-1">Job failed</p>
            <p>{wsError}</p>
          </GlassCard>
        )}

        {isDone && !wsError && (
          <GlassCard className="mb-6 px-4 py-3 text-sm text-[#10B981] border-[#10B981]/30">
            <p className="font-semibold mb-1">Processing complete</p>
            {resultKey ? <p className="text-xs font-mono">{resultKey}</p> : <p className="text-xs">Your video is ready.</p>}
          </GlassCard>
        )}

        {/* Pipeline stages */}
        <div className="space-y-3 mb-8">
          {PIPELINE_STAGES.map((s, i) => (
            <StageCard
              key={s.title}
              index={i}
              title={s.title}
              icon={s.icon}
              status={stageStatus(i, s.backendIdx, currentIdx, !!wsError)}
              subtitle={
                s.stageKey === stage && chunkInfo
                  ? `chunk ${chunkInfo.chunk} of ${chunkInfo.total}`
                  : undefined
              }
            />
          ))}
        </div>

        {/* Processing log */}
        <GlassCard className="p-4">
          <p className="text-xs font-semibold text-[#A0A0B0] uppercase tracking-wide mb-2">Processing log</p>
          <div className="h-40 overflow-y-auto font-mono text-xs space-y-1.5 pr-1">
            {log.map((line, i) => (
              <div key={i}>
                <span className="text-[#A78BFA]">[{line.time}]</span>{' '}
                <span className="text-[#F8F8FF]">{line.message}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
