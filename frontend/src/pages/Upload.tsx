import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiUploadVideo } from '../api/jobs'
import GlassCard from '../components/GlassCard'
import GlowButton from '../components/GlowButton'

const MAX_SIZE_BYTES = 2000 * 1024 * 1024  // 2 GB
const ACCEPTED = ['video/mp4', 'video/quicktime', 'video/webm']

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

export default function Upload() {
  const { isAuthenticated } = useAuthStore()
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const {
    file, uploadProgress, uploadStatus, s3Key, error,
    setFile, setProgress, setStatus, setS3Key, setError, reset,
  } = useUploadStore()

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login?redirect=/upload', { replace: true })
    }
  }, [isAuthenticated, navigate])

  const validateAndSet = (f: File) => {
    if (!ACCEPTED.includes(f.type)) {
      setError('Unsupported format. Please upload MP4, MOV, or WebM.')
      return
    }
    if (f.size > MAX_SIZE_BYTES) {
      setError(`File too large. Maximum size is 2 GB (got ${formatBytes(f.size)}).`)
      return
    }
    setFile(f)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) validateAndSet(f)
  }

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) validateAndSet(f)
  }

  const handleUpload = async () => {
    if (!file) return
    setStatus('uploading')
    setProgress(0)
    setError(null)

    try {
      const { s3_key } = await apiUploadVideo(file, setProgress)
      setS3Key(s3_key)
      setStatus('success')
      navigate('/editor')
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Upload failed. Please try again.'
      setError(msg)
    }
  }

  if (!isAuthenticated) return null

  return (
    <div className="min-h-screen bg-[#0A0A0F] py-16 px-4">
      <div className="max-w-2xl mx-auto">
        <p className="text-sm font-semibold text-[#A78BFA] mb-4 tracking-wide">UPLOAD</p>
        <h1 className="text-5xl font-black tracking-tight mb-2 bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent">
          Upload Your Video
        </h1>
        <p className="text-[#A0A0B0] mb-10">Supports MP4, MOV, WebM up to 2GB</p>

        <AnimatePresence mode="wait">
          {uploadStatus === 'success' ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <GlassCard glow className="p-8 text-center shadow-[0_0_40px_rgba(16,185,129,0.15)] border-[#10B981]/30">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 15 }}
                  className="w-14 h-14 rounded-full bg-[#10B981]/15 flex items-center justify-center mx-auto mb-4"
                >
                  <svg className="w-7 h-7 text-[#10B981]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                  </svg>
                </motion.div>
                <h2 className="text-xl font-black text-[#F8F8FF] mb-2">Video uploaded successfully</h2>
                <p className="text-[#A0A0B0] text-sm mb-2">Redirecting to the editor…</p>
                {s3Key && (
                  <p className="text-xs text-[#A0A0B0] font-mono bg-white/5 rounded-lg px-3 py-2 inline-block mt-2">
                    {s3Key}
                  </p>
                )}
                <div>
                  <button
                    onClick={reset}
                    className="mt-6 px-5 py-2.5 border border-white/10 rounded-xl text-sm font-semibold text-[#A0A0B0] hover:text-white hover:bg-white/5 transition-colors"
                  >
                    Upload another
                  </button>
                </div>
              </GlassCard>
            </motion.div>
          ) : file ? (
            <motion.div key="preview" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <GlassCard className="p-6">
                {/* Preview */}
                <div className="relative rounded-xl overflow-hidden bg-black mb-5 aspect-video">
                  <video
                    src={URL.createObjectURL(file)}
                    className="w-full h-full object-contain"
                    muted
                  />
                </div>

                <div className="flex items-start justify-between mb-6">
                  <div>
                    <div className="font-bold text-[#F8F8FF] text-sm mb-1">{file.name}</div>
                    <div className="flex items-center gap-2 text-xs text-[#A0A0B0]">
                      <span>{formatBytes(file.size)}</span>
                      <span className="px-2 py-0.5 rounded-full bg-[#7C3AED]/15 text-[#A78BFA] font-mono">
                        {file.type.split('/')[1]?.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <button onClick={reset} className="p-1.5 text-[#A0A0B0] hover:text-white transition-colors" title="Remove">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {uploadStatus === 'uploading' && (
                  <div className="mb-5">
                    <div className="flex justify-between text-xs text-[#A0A0B0] mb-1.5">
                      <span>Uploading…</span>
                      <span className="font-mono">{uploadProgress}%</span>
                    </div>
                    <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-[#7C3AED] to-[#2563EB] shadow-[0_0_10px_rgba(124,58,237,0.6)]"
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                )}

                {error && (
                  <div className="mb-4 bg-[#EF4444]/10 border border-[#EF4444]/30 text-[#EF4444] text-sm rounded-xl px-4 py-3">
                    {error}
                    <button
                      onClick={handleUpload}
                      className="ml-3 underline font-semibold"
                    >
                      Try again
                    </button>
                  </div>
                )}

                <GlowButton
                  onClick={handleUpload}
                  disabled={uploadStatus === 'uploading'}
                  className="w-full"
                >
                  {uploadStatus === 'uploading' ? 'Uploading…' : 'Upload'}
                </GlowButton>
              </GlassCard>
            </motion.div>
          ) : (
            <motion.div key="dropzone" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              {error && (
                <div className="mb-4 bg-[#EF4444]/10 border border-[#EF4444]/30 text-[#EF4444] text-sm rounded-xl px-4 py-3">
                  {error}
                </div>
              )}
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
                onDragLeave={() => setIsDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
                className={`backdrop-blur-xl rounded-2xl p-16 text-center cursor-pointer border-2 border-dashed transition-all duration-200 ${
                  isDragOver
                    ? 'border-[#7C3AED] bg-[#7C3AED]/10 shadow-[0_0_40px_rgba(124,58,237,0.25)]'
                    : 'border-[#7C3AED]/40 bg-white/5 pulse-border hover:border-[#7C3AED]/70 hover:bg-[#7C3AED]/5'
                }`}
              >
                <svg className="w-10 h-10 text-[#A78BFA] mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="font-semibold text-[#F8F8FF] mb-1">Drag & drop your video here</p>
                <div className="flex items-center gap-3 my-5 max-w-[160px] mx-auto">
                  <div className="flex-1 h-px bg-white/10" />
                  <span className="text-xs text-[#A0A0B0]">or</span>
                  <div className="flex-1 h-px bg-white/10" />
                </div>
                <span className="px-5 py-2.5 border border-[#7C3AED]/50 rounded-xl text-sm font-semibold text-[#F8F8FF] hover:bg-[#7C3AED]/10 transition-colors">
                  Browse Files
                </span>
                <p className="text-xs text-[#A0A0B0] mt-5">MP4, MOV, WebM — up to 2 GB</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                className="hidden"
                onChange={handleChange}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
