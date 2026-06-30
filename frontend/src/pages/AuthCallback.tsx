import { useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function AuthCallback() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { checkAuth } = useAuthStore()

  useEffect(() => {
    const token = params.get('token')
    if (!token) {
      navigate('/login', { replace: true })
      return
    }
    localStorage.setItem('ve_token', token)
    checkAuth().then(() => {
      navigate('/upload', { replace: true })
    })
  }, [])

  return (
    <div className="min-h-screen bg-[#FAFAF8] flex items-center justify-center">
      <div className="text-stone-500">Signing you in…</div>
    </div>
  )
}
