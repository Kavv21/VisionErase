import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiGetJobDownload } from '../api/jobs'
import GlassCard from '../components/GlassCard'
import GlowButton from '../components/GlowButton'

type QualityStatus = 'clean' | 'soft' | 'boundary' | 'flagged'

interface QualitySegment {
  status: QualityStatus
  label?: string
}

const QUALITY_COLORS: Record<QualityStatus, string> = {
  clean: 'bg-[#10B981]',
  soft: 'bg-yellow-400',
  boundary: 'bg-orange-400',
  flagged: 'bg-[#EF4444]',
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
        <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
          <div className="absolute inset-0 rounded-full bg-[#10B981] shadow-[0_0_12px_rgba(16,185,129,0.5)]" />
        </div>
      )}
      <p className="mt-2 text-xs font-semibold text-[#10B981]">
        {segments && segments.length > 0 ? 'Per-chunk quality' : 'Processing Complete'}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-[#A0A0B0]">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-[#10B981]" /> Clean
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-yellow-400" /> Soft
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-orange-400" /> Boundary
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-[#EF4444]" /> Flagged
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
    <div className="min-h-screen bg-[#0A0A0F] py-16 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="relative mb-2">
          <div className="absolute -inset-x-10 -inset-y-6 bg-[#10B981]/10 blur-3xl rounded-full pointer-events-none" />
          <h1 className="relative text-5xl font-black tracking-tight bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent">
            Your Video is Ready
          </h1>
        </div>
        <p className="text-[#A0A0B0] text-xs font-mono mb-10">{jobId}</p>

        {noResult ? (
          <GlassCard className="p-8 text-center border-dashed border-[#7C3AED]/40">
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4"
            >
              <svg className="w-6 h-6 text-[#A78BFA]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
              </svg>
            </motion.div>
            <h2 className="text-xl font-black text-[#F8F8FF] mb-2">No output was generated</h2>
            <p className="text-[#A0A0B0] text-sm mb-6">
              Processing completed but no output was generated. Please try again.
            </p>
            <GlowButton onClick={handleProcessAnother}>Try again</GlowButton>
          </GlassCard>
        ) : (
          <>
            {/* Video comparison */}
            <GlassCard className="p-6 mb-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <span className="inline-block px-3 py-1 rounded-full bg-white/10 text-xs font-semibold text-[#A0A0B0] mb-2">
                    Original
                  </span>
                  <div className="relative rounded-xl overflow-hidden bg-black aspect-video">
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
                      <div className="flex items-center justify-center h-full text-[#A0A0B0] text-xs px-4 text-center">
                        Original video unavailable (page was reloaded)
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <span className="inline-block px-3 py-1 rounded-full bg-[#7C3AED]/20 text-xs font-semibold text-[#A78BFA] mb-2">
                    Processed
                  </span>
                  <div
                    className="relative rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center"
                    style={{ boxShadow: '0 0 32px rgba(124,58,237,0.25)' }}
                  >
                    {downloadLoading && (
                      <div className="w-8 h-8 border-2 border-white/20 border-t-[#A78BFA] rounded-full animate-spin" />
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
                  className="w-10 h-10 flex items-center justify-center rounded-full bg-gradient-to-br from-[#7C3AED] to-[#2563EB] text-white shadow-[0_0_16px_rgba(124,58,237,0.4)] hover:shadow-[0_0_24px_rgba(124,58,237,0.6)] transition-shadow duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none shrink-0"
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
                <span className="text-xs font-mono text-[#A0A0B0] w-10">
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
                  className="flex-1 disabled:opacity-40"
                  style={{ accentColor: '#7C3AED' }}
                />
                <span className="text-xs font-mono text-[#A0A0B0] w-10">
                  {formatTime(duration)}
                </span>
              </div>
            </GlassCard>

            {/* Quality timeline */}
            <GlassCard className="p-6 mb-6">
              <p className="text-xs font-bold uppercase tracking-wide text-[#A0A0B0] mb-3">
                Quality timeline
              </p>
              <QualityTimeline />
            </GlassCard>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3">
              <GlowButton
                onClick={handleDownload}
                disabled={downloadLoading || !!noResult}
                className="inline-flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                </svg>
                Download Video
              </GlowButton>
              <GlowButton variant="outline" onClick={() => navigate('/editor')}>
                Edit Mask
              </GlowButton>
              <GlowButton variant="ghost" onClick={handleProcessAnother}>
                Process Another
              </GlowButton>
              <button
                onClick={() => navigate('/')}
                className="text-sm font-semibold text-[#A0A0B0] hover:text-white transition-colors underline"
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
