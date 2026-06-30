import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { apiGetJob } from '../api/jobs'

// Message shapes the /ws/{job_id} endpoint actually emits (see api/routers/websocket.py)
type WsMsg =
  | { type: 'progress'; stage: string; pct: number }
  | { type: 'complete'; result: { result_s3_key: string | null } }
  | { type: 'error'; message: string }
  // pub/sub forwarded messages from workers lack a `type` field
  | { stage: string; pct: number }

const STAGE_LABELS: Record<string, string> = {
  pending: 'Waiting to start',
  segmenting: 'Segmenting',
  tracking: 'Tracking',
  inpainting: 'Inpainting',
  stitching: 'Stitching',
  quality_check: 'Quality Check',
  completed: 'Complete',
  failed: 'Failed',
}

type ConnectionState = 'connecting' | 'open' | 'reconnecting' | 'failed'

export default function JobProgress() {
  const { jobId } = useParams<{ jobId: string }>()
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()

  const [connState, setConnState] = useState<ConnectionState>('connecting')
  const [stage, setStage] = useState('pending')
  const [pct, setPct] = useState(0)
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
            setStage(msg.stage ?? 'pending')
            setPct(typeof msg.pct === 'number' ? msg.pct : 0)
          } else if (msg.type === 'complete') {
            setStage('completed')
            setPct(100)
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
          setStage(msg.stage ?? 'pending')
          setPct(typeof msg.pct === 'number' ? msg.pct : 0)
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
          setPct(100)
          setResultKey(data.result_s3_key ?? null)
          setIsDone(true)
        } else if (data.status === 'failed') {
          setWsError(data.error ?? 'An error occurred')
          setIsDone(true)
        } else {
          setStage(data.status)
          setPct(data.progress_pct)
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

  if (!isAuthenticated) return null

  const stageLabel = STAGE_LABELS[stage] ?? stage

  return (
    <div className="min-h-screen bg-[#FAFAF8] py-16 px-4">
      <div className="max-w-2xl mx-auto">
        <p className="text-sm font-semibold text-red-600 mb-4">• Processing</p>
        <h1 className="text-4xl font-black text-stone-900 tracking-tight mb-2">Job Progress</h1>
        <p className="text-stone-400 text-xs font-mono mb-8">{jobId}</p>

        <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100">
          {/* Connection state */}
          {connState === 'connecting' && (
            <div className="flex items-center gap-2 text-stone-500 text-sm mb-6">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              Connecting…
            </div>
          )}
          {connState === 'reconnecting' && (
            <div className="flex items-center gap-2 text-stone-500 text-sm mb-6">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              Reconnecting…
            </div>
          )}
          {connState === 'failed' && !isDone && (
            <div className="mb-6 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-700">
              <p className="font-semibold mb-1">Connection lost</p>
              <p className="mb-3 text-xs">
                The server is not sending updates right now. This is expected while workers are
                starting up. The job is still queued and will process when a worker is available.
              </p>
              <button
                onClick={() => {
                  reconnectAttemptedRef.current = false
                  setConnState('connecting')
                  const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
                  const wsBase = apiBase.replace(/^http/, 'ws')
                  const ws = new WebSocket(`${wsBase}/ws/${jobId}`)
                  wsRef.current = ws
                  window.location.reload()
                }}
                className="text-xs font-semibold underline"
              >
                Refresh page
              </button>
            </div>
          )}

          {/* Stage + progress bar */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {!isDone && connState === 'open' && (
                  <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                )}
                {isDone && !wsError && (
                  <span className="w-2 h-2 rounded-full bg-green-400" />
                )}
                {wsError && (
                  <span className="w-2 h-2 rounded-full bg-red-400" />
                )}
                <span className="font-semibold text-stone-800 text-sm">{stageLabel}</span>
              </div>
              <span className="text-sm text-stone-500 font-mono">{Math.round(pct)}%</span>
            </div>
            <div className="h-2.5 bg-stone-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500 rounded-full transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {/* Error from WS */}
          {wsError && (
            <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
              <p className="font-semibold mb-1">Job failed</p>
              <p>{wsError}</p>
            </div>
          )}

          {/* Success */}
          {isDone && !wsError && (
            <div className="bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-sm text-green-700">
              <p className="font-semibold mb-1">Processing complete</p>
              {resultKey ? (
                <p className="text-xs font-mono">{resultKey}</p>
              ) : (
                <p className="text-xs">Your video is ready.</p>
              )}
            </div>
          )}

          {/* Pending/waiting explanation */}
          {!isDone && !wsError && stage === 'pending' && connState !== 'failed' && (
            <p className="text-xs text-stone-400 mt-4">
              Your job is queued. Progress will update automatically when a worker picks it up.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
