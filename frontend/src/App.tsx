import { useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Home from './pages/Home'
import About from './pages/About'
import Work from './pages/Work'
import Features from './pages/Features'
import Login from './pages/Login'
import Upload from './pages/Upload'
import Settings from './pages/Settings'
import AuthCallback from './pages/AuthCallback'
import MaskEditor from './pages/MaskEditor'
import JobProgress from './pages/JobProgress'
import { useAuthStore } from './store/authStore'

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
      <Route path="/work" element={<Work />} />
      <Route path="/features" element={<Features />} />
      <Route path="/login" element={<Login />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="/auth/callback" element={<AuthCallback />} />
      <Route path="/editor" element={<MaskEditor />} />
      <Route path="/progress/:jobId" element={<JobProgress />} />
    </Routes>
  )
}

export default function App() {
  const { checkAuth } = useAuthStore()

  useEffect(() => {
    checkAuth()
  }, [])

  return (
    <BrowserRouter>
      <Nav />
      <AppRoutes />
    </BrowserRouter>
  )
}
