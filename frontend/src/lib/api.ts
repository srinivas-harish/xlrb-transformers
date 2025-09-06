import axios from 'axios'
import type { RunDetail, RunRow } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const api = axios.create({ baseURL: API_BASE, timeout: 30000 })

export async function createRun(payload: any): Promise<{run_id:string, task_id:string, status:string}> {
  const { data } = await api.post('/runs', payload)
  return data
}

export async function listRuns(status?: string, limit = 50): Promise<RunRow[]> {
  const { data } = await api.get('/runs', { params: { status, limit } })
  return data
}

export async function getRun(id: string): Promise<RunDetail> {
  const { data } = await api.get(`/runs/${id}`)
  return data
}
