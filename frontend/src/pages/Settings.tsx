import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function Settings() {
  const { isAuthenticated, user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [displayName, setDisplayName] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login?redirect=/settings', { replace: true })
    }
    if (user?.display_name) setDisplayName(user.display_name)
  }, [isAuthenticated, user])

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleLogout = () => {
    logout()
    navigate('/', { replace: true })
  }

  if (!isAuthenticated) return null

  return (
    <div className="min-h-screen bg-[#FAFAF8] py-16 px-4">
      <div className="max-w-lg mx-auto">
        <p className="text-sm font-semibold text-red-600 mb-4">• Account</p>
        <h1 className="text-5xl font-black text-stone-900 tracking-tight mb-10">Settings</h1>

        <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100 mb-4">
          <h2 className="font-bold text-stone-900 mb-6">Profile</h2>

          <div className="mb-5">
            <label className="block text-xs font-semibold text-stone-600 mb-1">Email</label>
            <div className="text-sm text-stone-700 bg-stone-50 rounded-xl px-4 py-3 border border-stone-200">
              {user?.email ?? '—'}
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-xs font-semibold text-stone-600 mb-1">Display name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full border border-stone-200 rounded-xl px-4 py-3 text-sm outline-none focus:border-stone-400 transition-colors"
            />
          </div>

          <button
            onClick={handleSave}
            className="px-5 py-2.5 bg-stone-900 text-white font-semibold rounded-xl text-sm hover:bg-stone-800 transition-colors"
          >
            {saved ? 'Saved! (Coming soon)' : 'Save changes'}
          </button>
          {saved && <p className="text-xs text-stone-400 mt-2">Profile editing is coming soon — changes were not persisted.</p>}
        </div>

        <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100">
          <h2 className="font-bold text-stone-900 mb-4">Session</h2>
          <button
            onClick={handleLogout}
            className="px-5 py-2.5 border border-red-200 text-red-600 font-semibold rounded-xl text-sm hover:bg-red-50 transition-colors"
          >
            Log out
          </button>
        </div>
      </div>
    </div>
  )
}
