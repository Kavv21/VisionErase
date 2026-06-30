import { apiClient } from './client'

export interface UserProfile {
  id: string
  email: string
  display_name: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export async function apiRegister(
  email: string,
  password: string,
  display_name: string
): Promise<TokenResponse> {
  const { data } = await apiClient.post<TokenResponse>('/api/v1/auth/register', {
    email,
    password,
    display_name,
  })
  return data
}

export async function apiLogin(email: string, password: string): Promise<TokenResponse> {
  const { data } = await apiClient.post<TokenResponse>('/api/v1/auth/login', {
    email,
    password,
  })
  return data
}

export async function apiGetMe(): Promise<UserProfile> {
  const { data } = await apiClient.get<UserProfile>('/api/v1/auth/me')
  return data
}
