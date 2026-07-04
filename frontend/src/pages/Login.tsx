import { useState, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import GlassCard from '../components/GlassCard'
import GlowButton from '../components/GlowButton'
import { useAuthStore } from '../store/authStore'

export default function Login() {
  const { login, register, loginWithGoogle } = useAuthStore()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const redirectTo = params.get('redirect') ?? '/upload'

  const [mode, setMode] = useState<'signin' | 'register'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === 'signin') {
        await login(email, password)
      } else {
        if (!displayName.trim()) { setError('Display name is required'); setLoading(false); return }
        await register(email, password, displayName)
      }
      navigate(redirectTo, { replace: true })
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Something went wrong. Please try again.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const inputClass =
    'w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-[#F8F8FF] placeholder:text-[#A0A0B0]/60 outline-none focus:border-[#7C3AED] focus:ring-2 focus:ring-[#7C3AED]/30 transition-all duration-200'

  return (
    <div className="min-h-screen bg-[#0A0A0F] flex items-center justify-center px-4 relative overflow-hidden">
      <div className="orb orb-purple w-[420px] h-[420px] -top-20 -left-20" />
      <div className="orb orb-blue w-[380px] h-[380px] bottom-0 -right-20" />

      <div className="relative w-full max-w-md">
        <div className="text-center mb-8">
          <p className="text-xl font-black bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent mb-4">
            VisionErase
          </p>
          <h1 className="text-3xl font-black text-[#F8F8FF] tracking-tight mb-2">
            {mode === 'signin' ? 'Sign in to continue' : 'Create an account'}
          </h1>
          <p className="text-[#A0A0B0] text-sm">
            {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={() => { setMode(mode === 'signin' ? 'register' : 'signin'); setError(null) }}
              className="text-[#A78BFA] font-semibold hover:underline"
            >
              {mode === 'signin' ? 'Sign up' : 'Sign in'}
            </button>
          </p>
        </div>

        <GlassCard className="p-8">
          {/* Google */}
          <button
            onClick={loginWithGoogle}
            className="w-full flex items-center justify-center gap-3 bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-sm font-semibold text-[#F8F8FF] hover:border-white/30 hover:bg-white/10 transition-all duration-200 mb-6"
          >
            <svg className="w-5 h-5" viewBox="0 0 48 48">
              <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.2-.4-4.7H24v8.9h12.7c-.5 2.8-2.2 5.2-4.7 6.8v5.6h7.6c4.4-4.1 7-10.1 7-16.6z"/>
              <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.6-5.6c-2.2 1.4-4.9 2.2-8.3 2.2-6.4 0-11.8-4.3-13.7-10.1H2.4v5.8C6.4 42.7 14.6 48 24 48z"/>
              <path fill="#FBBC05" d="M10.3 28.7c-.5-1.4-.8-2.9-.8-4.7s.3-3.3.8-4.7v-5.8H2.4C.9 16.8 0 20.3 0 24s.9 7.2 2.4 10.5l7.9-5.8z"/>
              <path fill="#EA4335" d="M24 9.5c3.6 0 6.8 1.2 9.4 3.6l7-7C36 2.1 30.5 0 24 0 14.6 0 6.4 5.3 2.4 13.5l7.9 5.8C12.2 13.8 17.6 9.5 24 9.5z"/>
            </svg>
            Continue with Google
          </button>

          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/10" />
            </div>
            <div className="relative flex justify-center">
              <span className="px-3 bg-[#12121A] text-xs text-[#A0A0B0]">or continue with email</span>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <div>
                <label className="block text-xs font-semibold text-[#A0A0B0] mb-1">Display name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Your name"
                  className={inputClass}
                  required
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-semibold text-[#A0A0B0] mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className={inputClass}
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-[#A0A0B0] mb-1">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === 'register' ? 'At least 8 characters' : '••••••••'}
                  minLength={mode === 'register' ? 8 : undefined}
                  className={`${inputClass} pr-11`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#A0A0B0] hover:text-white transition-colors"
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.243 4.243L9.88 9.88" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="bg-[#EF4444]/10 border border-[#EF4444]/30 text-[#EF4444] text-sm rounded-xl px-4 py-3">
                {error}
              </div>
            )}

            <GlowButton type="submit" disabled={loading} className="w-full">
              {loading ? 'Please wait…' : mode === 'signin' ? 'Sign in' : 'Create account'}
            </GlowButton>
          </form>
        </GlassCard>
      </div>
    </div>
  )
}
