export type RunStatus = 'QUEUED'|'RUNNING'|'COMPLETE'|'FAILED'|'STARTED'

export type EpochLog = {
  epoch: number
  time_sec?: number
  train_acc?: number
  train_loss_ema?: number
  val_acc?: number
  val_f1_macro?: number
  lr?: number
  gates?: {
    alpha_hdim?: { mean?: number; std?: number; min?: number; max?: number; shape?: number[] }
    alpha_attn?: { mean?: number; std?: number; min?: number; max?: number; shape?: number[] }
  }
}

export type ResultBlob = {
  seed: number
  device: string
  cfg: Record<string, any>
  param_counts: Record<string, number>
  best: { val_acc: number; epoch: number }
  epochs: EpochLog[]
  save_dir?: string | null
  profiling?: {
    nsys_summary?: {
      gpu_busy_pct?: number | null
      total_gpu_time_ms?: number | null
      memcpy_time_ms?: number | null
      memset_time_ms?: number | null
      transfer_overhead_pct?: number | null
      num_unique_kernels?: number | null
    }
    nsys_details?: {
      kernels?: Array<{ name: string; calls?: number | null; time_ms?: number | null; avg_ms?: number | null; pct_gpu_time?: number | null }>
      cuda_api?: Array<{ name: string; calls?: number | null; time_ms?: number | null; avg_ms?: number | null }>
      transfers?: Array<{ name: string; calls?: number | null; time_ms?: number | null }>
    }
    debug?: Record<string, any>
  }
}

export type RunRow = {
  id: string
  task_id?: string | null
  status: RunStatus
  ablation?: string | null
  overrides?: Record<string, any> | null
  epochs: number
  batch_size: number
  max_len: number
  device?: string | null
  save_dir?: string | null
  save_artifacts?: boolean
  early_stop?: any
  best_val_acc?: number | null
  created_at: string
  updated_at: string
}

export type RunDetail = RunRow & {
  result?: ResultBlob | null
  epochs_rows?: Array<{
    epoch: number
    time_sec?: number
    train_acc?: number
    val_acc?: number
    val_f1_macro?: number
    lr?: number
  }>
  artifacts?: Array<{ id: string; path: string; kind?: string; size?: number; created_at?: string }>
  error?: string | null
}

export type KernelSanity = {
  ok: boolean
  env?: { is_wsl?: boolean }
  ncu: { found: boolean; version?: string | null; path?: string | null }
  nsys?: { found: boolean }
  python: string
  torch_cuda_available?: boolean | null
  artifact?: string | null
  kernels: Array<{ name: string; calls?: number | null; time_ms?: number | null; avg_ms?: number | null; pct_gpu_time?: number | null }>
  num_unique_kernels: number
  tried_sets?: string[][]
  stdout?: string | null
  stderr?: string | null
  raw_preview?: string | null
  notes?: string[]
}
