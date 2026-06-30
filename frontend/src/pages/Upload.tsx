import { useEffect, useRef, type DragEvent, type ChangeEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useUploadStore } from '../store/uploadStore'
import { apiUploadVideo } from '../api/jobs'

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
    <div className="min-h-screen bg-[#FAFAF8] py-16 px-4">
      <div className="max-w-2xl mx-auto">
        <p className="text-sm font-semibold text-red-600 mb-4">• Upload</p>
        <h1 className="text-5xl font-black text-stone-900 tracking-tight mb-2">
          Upload your video
        </h1>
        <p className="text-stone-500 mb-10">We'll remove the object you mark from every frame.</p>

        {uploadStatus === 'success' ? (
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100 text-center">
            <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-7 h-7 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h2 className="text-xl font-black text-stone-900 mb-2">Video uploaded</h2>
            <p className="text-stone-500 text-sm mb-2">Ready for the next step.</p>
            {s3Key && (
              <p className="text-xs text-stone-400 font-mono bg-stone-50 rounded-lg px-3 py-2 inline-block mt-2">
                {s3Key}
              </p>
            )}
            <button
              onClick={reset}
              className="mt-6 px-5 py-2.5 border border-stone-200 rounded-xl text-sm font-semibold text-stone-600 hover:bg-stone-50 transition-colors"
            >
              Upload another
            </button>
          </div>
        ) : file ? (
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100">
            {/* Preview */}
            <div className="relative rounded-xl overflow-hidden bg-stone-900 mb-5 aspect-video">
              <video
                src={URL.createObjectURL(file)}
                className="w-full h-full object-contain"
                muted
              />
            </div>

            <div className="flex items-start justify-between mb-6">
              <div>
                <div className="font-bold text-stone-900 text-sm mb-0.5">{file.name}</div>
                <div className="text-xs text-stone-400">{formatBytes(file.size)}</div>
              </div>
              <button onClick={reset} className="p-1.5 text-stone-400 hover:text-stone-700 transition-colors" title="Remove">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {uploadStatus === 'uploading' && (
              <div className="mb-5">
                <div className="flex justify-between text-xs text-stone-500 mb-1.5">
                  <span>Uploading…</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="h-2 bg-stone-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {error && (
              <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
                {error}
                <button
                  onClick={handleUpload}
                  className="ml-3 underline font-semibold"
                >
                  Try again
                </button>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={uploadStatus === 'uploading'}
              className="w-full py-3 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploadStatus === 'uploading' ? 'Uploading…' : 'Upload'}
            </button>
          </div>
        ) : (
          <div>
            {error && (
              <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
                {error}
              </div>
            )}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-stone-300 rounded-2xl p-16 text-center cursor-pointer hover:border-red-400 hover:bg-red-50/30 transition-all duration-150"
            >
              <svg className="w-10 h-10 text-stone-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="font-semibold text-stone-700 mb-1">Drag and drop your video here</p>
              <p className="text-sm text-stone-400 mb-5">or</p>
              <span className="px-5 py-2.5 border border-stone-300 rounded-xl text-sm font-semibold text-stone-600 hover:bg-white transition-colors">
                Browse files
              </span>
              <p className="text-xs text-stone-400 mt-5">MP4, MOV, WebM — up to 2 GB</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm"
              className="hidden"
              onChange={handleChange}
            />
          </div>
        )}
      </div>
    </div>
  )
}
