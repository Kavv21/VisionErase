import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function Nav() {
  const { isAuthenticated, user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const [avatarOpen, setAvatarOpen] = useState(false)

  const initials = user?.display_name
    ? user.display_name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()
    : '?'

  const handleLogout = () => {
    logout()
    setAvatarOpen(false)
    navigate('/')
  }

  return (
    <nav className="sticky top-0 z-50 bg-[#FAFAF8]/95 backdrop-blur border-b border-stone-200">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="text-xl font-bold text-stone-900 tracking-tight">
          VisionErase
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8 text-sm text-stone-600 font-medium">
          <Link to="/about" className="hover:text-stone-900 transition-colors">About</Link>
          <Link to="/work" className="hover:text-stone-900 transition-colors">Work</Link>
          <Link to="/features" className="hover:text-stone-900 transition-colors">Features</Link>
        </div>

        {/* Desktop right side */}
        <div className="hidden md:flex items-center gap-3">
          {isAuthenticated ? (
            <div className="relative">
              <button
                onClick={() => setAvatarOpen((o) => !o)}
                className="w-9 h-9 rounded-full bg-red-600 text-white text-sm font-semibold flex items-center justify-center hover:bg-red-700 transition-colors"
              >
                {initials}
              </button>
              {avatarOpen && (
                <div className="absolute right-0 top-11 w-44 bg-white rounded-xl shadow-lg border border-stone-200 py-1 z-50">
                  <Link
                    to="/settings"
                    onClick={() => setAvatarOpen(false)}
                    className="block px-4 py-2.5 text-sm text-stone-700 hover:bg-stone-50"
                  >
                    Settings
                  </Link>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2.5 text-sm text-stone-700 hover:bg-stone-50"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          ) : (
            <Link
              to="/login"
              className="px-4 py-2 bg-red-600 text-white text-sm font-semibold rounded-lg hover:bg-red-700 transition-colors"
            >
              Login
            </Link>
          )}
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-2 text-stone-600"
          onClick={() => setMenuOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          {menuOpen ? (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden bg-white border-t border-stone-200 px-6 py-4 flex flex-col gap-4">
          <Link to="/about" className="text-stone-700 font-medium" onClick={() => setMenuOpen(false)}>About</Link>
          <Link to="/work" className="text-stone-700 font-medium" onClick={() => setMenuOpen(false)}>Work</Link>
          <Link to="/features" className="text-stone-700 font-medium" onClick={() => setMenuOpen(false)}>Features</Link>
          {isAuthenticated ? (
            <>
              <Link to="/settings" className="text-stone-700 font-medium" onClick={() => setMenuOpen(false)}>Settings</Link>
              <button onClick={() => { handleLogout(); setMenuOpen(false) }} className="text-left text-stone-700 font-medium">Logout</button>
            </>
          ) : (
            <Link
              to="/login"
              className="px-4 py-2 bg-red-600 text-white text-sm font-semibold rounded-lg text-center"
              onClick={() => setMenuOpen(false)}
            >
              Login
            </Link>
          )}
        </div>
      )}
    </nav>
  )
}
