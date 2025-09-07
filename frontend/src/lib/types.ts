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
