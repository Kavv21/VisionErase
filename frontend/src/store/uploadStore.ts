import { create } from 'zustand'

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

interface UploadState {
  file: File | null
  uploadProgress: number
  uploadStatus: UploadStatus
  jobId: string | null
  s3Key: string | null
  error: string | null

  setFile: (file: File | null) => void
  setProgress: (pct: number) => void
  setStatus: (status: UploadStatus) => void
  setS3Key: (key: string) => void
  setJobId: (id: string) => void
  setError: (msg: string | null) => void
  reset: () => void
}

export const useUploadStore = create<UploadState>((set) => ({
  file: null,
  uploadProgress: 0,
  uploadStatus: 'idle',
  jobId: null,
  s3Key: null,
  error: null,

  setFile: (file) => set({ file, uploadStatus: 'idle', error: null, uploadProgress: 0 }),
  setProgress: (pct) => set({ uploadProgress: pct }),
  setStatus: (status) => set({ uploadStatus: status }),
  setS3Key: (key) => set({ s3Key: key }),
  setJobId: (id) => set({ jobId: id }),
  setError: (msg) => set({ error: msg, uploadStatus: 'error' }),
  reset: () =>
    set({
      file: null,
      uploadProgress: 0,
      uploadStatus: 'idle',
      jobId: null,
      s3Key: null,
      error: null,
    }),
}))
