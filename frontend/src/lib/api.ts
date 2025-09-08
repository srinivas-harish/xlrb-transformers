import axios from 'axios'
import type { RunDetail, RunRow, KernelSanity } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const api = axios.create({ baseURL: API_BASE, timeout: 30000 })

export async function createRun(payload: any): Promise<{run_id:string, task_id:string, status:string}> {
  const { data } = await api.post('/runs', payload)
  return data
}

export async function listRuns(status?: string, limit = 50): Promise<RunRow[]> {
  const { data } = await api.get('/runs', { params: { status, limit } })
  // Map backend fields (run_id, etc.) to frontend types
  return (data as any[]).map((r: any) => ({
    id: r.run_id,
    task_id: r.task_id ?? null,
    status: r.status,
    ablation: r.ablation ?? null,
    overrides: r.overrides ?? {},
    epochs: r.epochs,
    batch_size: r.batch_size,
    max_len: r.max_len,
    device: r.device ?? 'auto',
    save_dir: r.save_dir ?? null,
    save_artifacts: !!r.save_artifacts,
    early_stop: r.early_stop ?? null,
    best_val_acc: r.best_val_acc ?? null,
    created_at: r.created_at,
    updated_at: r.updated_at,
  }))
}

export async function getRun(id: string): Promise<RunDetail> {
  const { data } = await api.get(`/runs/${id}`)
  const r = data as any
  const artifacts = (r.artifacts || []).map((a: any) => ({
    id: a.path, // backend doesn't return id; use path as stable key
    path: a.path,
    kind: a.kind,
    size: typeof a.bytes === 'number' ? a.bytes : undefined,
  }))
  const detail: RunDetail = {
    id: r.run_id,
    task_id: r.task_id ?? null,
    status: r.status,
    ablation: r.ablation ?? null,
    overrides: r.overrides ?? {},
    epochs: r.epochs,
    batch_size: r.batch_size,
    max_len: r.max_len,
    device: r.device ?? 'auto',
    save_dir: r.save_dir ?? null,
    save_artifacts: !!r.save_artifacts,
    early_stop: r.early_stop ?? null,
    best_val_acc: r.best_val_acc ?? null,
    created_at: r.created_at,
    updated_at: r.updated_at,
    result: r.result ?? null,
    epochs_rows: (r.epochs_log || []).map((e: any) => ({
      epoch: e.epoch,
      time_sec: e.time_sec,
      train_acc: e.train_acc,
      val_acc: e.val_acc,
      val_f1_macro: e.val_f1_macro,
      lr: e.lr,
    })),
    artifacts,
    error: r.error ?? null,
  }
  return detail
}

export async function refreshProfiling(id: string): Promise<any> {
  const { data } = await api.post(`/runs/${id}/refresh_profiling`)
  return data
}

export async function kernelSanity(): Promise<KernelSanity> {
  const { data } = await api.post('/diag/kernel_test')
  return data as KernelSanity
}
