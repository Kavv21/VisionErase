import { apiClient } from './client'

export interface UploadUrlResponse {
  upload_url: string
  s3_key: string
  expires_in: number
}

export interface MaskPoint {
  x: number
  y: number
}

export interface MaskData {
  points: MaskPoint[]
  frame_index: number
}

export interface SegmentPreviewResponse {
  mask_points: MaskPoint[]
  stub: boolean
}

export interface CreateJobResponse {
  job_id: string
  status: string
  progress_pct: number
  created_at: string
  result_s3_key: string | null
  error: string | null
  cached: boolean
}

export async function apiRequestUploadUrl(
  filename: string,
  contentType: string,
  sizeBytes: number
): Promise<UploadUrlResponse> {
  const { data } = await apiClient.post<UploadUrlResponse>('/api/v1/jobs/upload-url', {
    filename,
    content_type: contentType,
    size_bytes: sizeBytes,
  })
  return data
}

export interface VideoUploadResponse {
  s3_key: string
  size_bytes: number
}

export async function apiUploadVideo(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<VideoUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<VideoUploadResponse>('/api/v1/jobs/upload', form, {
  headers: {
    'Content-Type': undefined,  // let browser set it with correct boundary
  },
  onUploadProgress: (ev) => {
    if (onProgress && ev.total) onProgress(Math.round((ev.loaded / ev.total) * 100))
  },
})
  return data
}

export async function apiSegmentPreview(
  videoS3Key: string,
  point: MaskPoint,
  frameIndex = 0
): Promise<SegmentPreviewResponse> {
  const { data } = await apiClient.post<SegmentPreviewResponse>('/api/v1/segment/preview', {
    video_s3_key: videoS3Key,
    point,
    frame_index: frameIndex,
  })
  return data
}

export async function apiCreateJob(
  videoS3Key: string,
  mask: MaskData
): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>('/api/v1/jobs/', {
    video_s3_key: videoS3Key,
    mask,
  })
  return data
}
