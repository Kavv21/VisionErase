import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
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
    <nav className="sticky top-0 z-50 backdrop-blur-xl bg-black/40 border-b border-white/10">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link
          to="/"
          className="text-xl font-black tracking-tight bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent"
        >
          VisionErase
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-8 text-sm text-[#A0A0B0] font-medium">
          <Link to="/about" className="hover:text-white transition-colors duration-200">About</Link>
          <Link to="/work" className="hover:text-white transition-colors duration-200">Work</Link>
          <Link to="/features" className="hover:text-white transition-colors duration-200">Features</Link>
        </div>

        {/* Desktop right side */}
        <div className="hidden md:flex items-center gap-3">
          {isAuthenticated ? (
            <div className="relative">
              <button
                onClick={() => setAvatarOpen((o) => !o)}
                className="w-9 h-9 rounded-full bg-gradient-to-br from-[#7C3AED] to-[#2563EB] text-white text-sm font-semibold flex items-center justify-center shadow-[0_0_16px_rgba(124,58,237,0.4)] hover:shadow-[0_0_24px_rgba(124,58,237,0.6)] transition-shadow duration-200"
              >
                {initials}
              </button>
              <AnimatePresence>
                {avatarOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-11 w-44 backdrop-blur-xl bg-[#12121A]/95 rounded-xl shadow-xl border border-white/10 py-1 z-50"
                  >
                    <Link
                      to="/settings"
                      onClick={() => setAvatarOpen(false)}
                      className="block px-4 py-2.5 text-sm text-[#A0A0B0] hover:text-white hover:bg-white/5 transition-colors"
                    >
                      Settings
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="w-full text-left px-4 py-2.5 text-sm text-[#A0A0B0] hover:text-white hover:bg-white/5 transition-colors"
                    >
                      Logout
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <Link
              to="/login"
              className="px-4 py-2 bg-gradient-to-r from-[#7C3AED] to-[#2563EB] text-white text-sm font-semibold rounded-lg shadow-[0_0_16px_rgba(124,58,237,0.35)] hover:shadow-[0_0_24px_rgba(124,58,237,0.55)] transition-shadow duration-200"
            >
              Login
            </Link>
          )}
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-2 text-[#A0A0B0]"
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
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden fixed inset-0 top-16 bg-[#0A0A0F]/98 backdrop-blur-xl px-6 py-8 flex flex-col gap-6 text-lg"
          >
            <Link to="/about" className="text-[#A0A0B0] hover:text-white font-medium transition-colors" onClick={() => setMenuOpen(false)}>About</Link>
            <Link to="/work" className="text-[#A0A0B0] hover:text-white font-medium transition-colors" onClick={() => setMenuOpen(false)}>Work</Link>
            <Link to="/features" className="text-[#A0A0B0] hover:text-white font-medium transition-colors" onClick={() => setMenuOpen(false)}>Features</Link>
            {isAuthenticated ? (
              <>
                <Link to="/settings" className="text-[#A0A0B0] hover:text-white font-medium transition-colors" onClick={() => setMenuOpen(false)}>Settings</Link>
                <button onClick={() => { handleLogout(); setMenuOpen(false) }} className="text-left text-[#A0A0B0] hover:text-white font-medium transition-colors">Logout</button>
              </>
            ) : (
              <Link
                to="/login"
                className="px-4 py-3 bg-gradient-to-r from-[#7C3AED] to-[#2563EB] text-white font-semibold rounded-lg text-center"
                onClick={() => setMenuOpen(false)}
              >
                Login
              </Link>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
