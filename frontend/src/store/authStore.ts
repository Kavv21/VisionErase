import { create } from 'zustand'
import { apiLogin, apiRegister, apiGetMe, type UserProfile } from '../api/auth'

interface AuthState {
  user: UserProfile | null
  token: string | null
  isAuthenticated: boolean

  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string) => Promise<void>
  logout: () => void
  loginWithGoogle: () => void
  checkAuth: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('ve_token'),
  isAuthenticated: !!localStorage.getItem('ve_token'),

  login: async (email, password) => {
    const { access_token } = await apiLogin(email, password)
    localStorage.setItem('ve_token', access_token)
    const user = await apiGetMe()
    set({ token: access_token, user, isAuthenticated: true })
  },

  register: async (email, password, displayName) => {
    const { access_token } = await apiRegister(email, password, displayName)
    localStorage.setItem('ve_token', access_token)
    const user = await apiGetMe()
    set({ token: access_token, user, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('ve_token')
    set({ token: null, user: null, isAuthenticated: false })
  },

  loginWithGoogle: () => {
    window.location.href = `${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/api/v1/auth/google`
  },

  checkAuth: async () => {
    const token = localStorage.getItem('ve_token')
    if (!token) {
      set({ isAuthenticated: false, user: null })
      return
    }
    try {
      const user = await apiGetMe()
      set({ token, user, isAuthenticated: true })
    } catch {
      localStorage.removeItem('ve_token')
      set({ token: null, user: null, isAuthenticated: false })
    }
  },
}))
