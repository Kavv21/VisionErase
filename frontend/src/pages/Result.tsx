import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiGetJobDownload } from '../api/jobs'

type QualityStatus = 'clean' | 'soft' | 'boundary' | 'flagged'

interface QualitySegment {
  status: QualityStatus
  label?: string
}

const QUALITY_COLORS: Record<QualityStatus, string> = {
  clean: 'bg-green-500',
  soft: 'bg-yellow-400',
  boundary: 'bg-orange-400',
  flagged: 'bg-red-500',
}

function QualityTimeline({ segments }: { segments?: QualitySegment[] }) {
  return (
    <div>
      {segments && segments.length > 0 ? (
        <div className="flex h-3 w-full overflow-hidden rounded-full">
          {segments.map((seg, i) => (
            <div key={i} className={`flex-1 ${QUALITY_COLORS[seg.status]}`} title={seg.label} />
          ))}
        </div>
      ) : (
        <div className="relative h-3 w-full overflow-hidden rounded-full bg-stone-100">
          <div className="absolute inset-0 rounded-full bg-green-500" />
        </div>
      )}
      <p className="mt-2 text-xs font-semibold text-stone-500">
        {segments && segments.length > 0 ? 'Per-chunk quality' : 'Processing complete'}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-stone-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-green-500" /> Clean
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-yellow-400" /> Soft
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-orange-400" /> Boundary
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-500" /> Flagged
        </span>
      </div>
    </div>
  )
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function Result() {
  const { jobId } = useParams<{ jobId: string }>()
  const { isAuthenticated } = useAuthStore()
  const { file, reset } = useUploadStore()
  const navigate = useNavigate()

  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [downloadLoading, setDownloadLoading] = useState(true)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [noResult, setNoResult] = useState(false)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  const originalRef = useRef<HTMLVideoElement>(null)
  const processedRef = useRef<HTMLVideoElement>(null)

  const [originalUrl, setOriginalUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setOriginalUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setOriginalUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!isAuthenticated) {
      navigate(`/login?redirect=/result/${jobId}`, { replace: true })
    }
  }, [isAuthenticated, jobId, navigate])

  useEffect(() => {
    if (!jobId || !isAuthenticated) return

    let cancelled = false
    setDownloadLoading(true)
    setDownloadError(null)
    setNoResult(false)

    apiGetJobDownload(jobId)
      .then((data) => {
        if (cancelled) return
        setDownloadUrl(data.download_url)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        const detail =
          (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        if (detail === 'No result available') {
          setNoResult(true)
        } else {
          setDownloadError(detail ?? 'Could not load the processed video.')
        }
      })
      .finally(() => {
        if (!cancelled) setDownloadLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [jobId, isAuthenticated])

  if (!isAuthenticated) return null

  const handleLoadedMetadata = () => {
    const dur = originalRef.current?.duration
    if (dur && Number.isFinite(dur)) setDuration(dur)
  }

  const handleTimeUpdate = () => {
    const original = originalRef.current
    const processed = processedRef.current
    if (!original) return
    setCurrentTime(original.currentTime)
    if (processed && Math.abs(processed.currentTime - original.currentTime) > 0.25) {
      processed.currentTime = original.currentTime
    }
  }

  const togglePlayPause = () => {
    const original = originalRef.current
    const processed = processedRef.current
    if (!original) return
    if (isPlaying) {
      original.pause()
      processed?.pause()
      setIsPlaying(false)
    } else {
      void original.play()
      void processed?.play()
      setIsPlaying(true)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = Number(e.target.value)
    const original = originalRef.current
    const processed = processedRef.current
    if (original) original.currentTime = t
    if (processed) processed.currentTime = t
    setCurrentTime(t)
  }

  const handleDownload = async () => {
    if (!jobId) return
    try {
      const { download_url } = await apiGetJobDownload(jobId)
      const a = document.createElement('a')
      a.href = download_url
      a.download = ''
      document.body.appendChild(a)
      a.click()
      a.remove()
    } catch {
      setDownloadError('Could not start the download. Please try again.')
    }
  }

  const handleProcessAnother = () => {
    reset()
    navigate('/upload')
  }

  return (
    <div className="min-h-screen bg-[#FAFAF8] py-16 px-4">
      <div className="max-w-4xl mx-auto">
        <p className="text-sm font-semibold text-red-600 mb-4">• Result</p>
        <h1 className="text-5xl font-black text-stone-900 tracking-tight mb-2">
          Your video is ready
        </h1>
        <p className="text-stone-400 text-xs font-mono mb-10">{jobId}</p>

        {noResult ? (
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100 text-center">
            <h2 className="text-xl font-black text-stone-900 mb-2">No output was generated</h2>
            <p className="text-stone-500 text-sm mb-6">
              Processing completed but no output was generated. Please try again.
            </p>
            <button
              onClick={handleProcessAnother}
              className="px-5 py-2.5 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-colors"
            >
              Try again
            </button>
          </div>
        ) : (
          <>
            {/* Video comparison */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100 mb-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-bold uppercase tracking-wide text-stone-400 mb-2">
                    Original
                  </p>
                  <div className="relative rounded-xl overflow-hidden bg-stone-900 aspect-video">
                    {originalUrl ? (
                      <video
                        ref={originalRef}
                        src={originalUrl}
                        className="w-full h-full object-contain"
                        muted
                        onLoadedMetadata={handleLoadedMetadata}
                        onTimeUpdate={handleTimeUpdate}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-stone-400 text-xs px-4 text-center">
                        Original video unavailable (page was reloaded)
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <p className="text-xs font-bold uppercase tracking-wide text-stone-400 mb-2">
                    Processed
                  </p>
                  <div className="relative rounded-xl overflow-hidden bg-stone-900 aspect-video flex items-center justify-center">
                    {downloadLoading && (
                      <div className="w-8 h-8 border-2 border-stone-600 border-t-white rounded-full animate-spin" />
                    )}
                    {!downloadLoading && downloadError && (
                      <p className="text-red-300 text-xs px-4 text-center">{downloadError}</p>
                    )}
                    {!downloadLoading && downloadUrl && (
                      <video
                        ref={processedRef}
                        src={downloadUrl}
                        className="w-full h-full object-contain"
                        muted
                      />
                    )}
                  </div>
                </div>
              </div>

              {/* Shared controls */}
              <div className="mt-5 flex items-center gap-4">
                <button
                  onClick={togglePlayPause}
                  disabled={!originalUrl}
                  className="w-10 h-10 flex items-center justify-center rounded-full bg-red-600 text-white hover:bg-red-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                >
                  {isPlaying ? (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="5" width="4" height="14" />
                      <rect x="14" y="5" width="4" height="14" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                </button>
                <span className="text-xs font-mono text-stone-400 w-10">
                  {formatTime(currentTime)}
                </span>
                <input
                  type="range"
                  min={0}
                  max={duration || 0}
                  step={0.01}
                  value={currentTime}
                  onChange={handleSeek}
                  disabled={!originalUrl || !duration}
                  className="flex-1 accent-red-600 disabled:opacity-40"
                />
                <span className="text-xs font-mono text-stone-400 w-10">
                  {formatTime(duration)}
                </span>
              </div>
            </div>

            {/* Quality timeline */}
            <div className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100 mb-6">
              <p className="text-xs font-bold uppercase tracking-wide text-stone-400 mb-3">
                Quality timeline
              </p>
              <QualityTimeline />
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={handleDownload}
                disabled={downloadLoading || !!noResult}
                className="px-5 py-2.5 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Download
              </button>
              <button
                onClick={handleProcessAnother}
                className="px-5 py-2.5 border border-stone-300 rounded-xl text-sm font-semibold text-stone-600 hover:bg-white transition-colors"
              >
                Process another video
              </button>
              <button
                onClick={() => navigate('/')}
                className="text-sm font-semibold text-stone-500 hover:text-stone-800 transition-colors underline"
              >
                Back to dashboard
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
