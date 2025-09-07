import { useEffect, useMemo, useState } from 'react'
import JSONEditor from './JSONEditor'
import { createRun } from '../lib/api'

const ABLATIONS = [
  'A1_QKV_WVbridge','A2_QKV_TargetW','A3_QKV_only','A4_HDIM_only','A5_HDIM_bilinear',
  'A6_HDIM_proj32_mlp128','A7_HDIM_GELU','A8_Router_CLS','A9_Router_Temp0p5','A10_Inject_Last2',
  'H1_proj32_mlp128','H2_proj16_mlp64','H3_gate0p10','H4_dropout0','H5_routepool_cls',
  'H6_last6','H7_router_temp0p5','H8_pair_bilinear','H9_val_concat_only','H10_share_proj',
  'H11_token_temp0p8','H12_pair_cosine','H13_cosine_concat_only','H14_gate0p10_drop0p05','H15_last3',
  'H16_entreg_1e3','H17_proj_LN','H18_val_residual','H19_route_dim64','H20_proj32_concat_only'
]

export default function RunForm({ onSubmitted }: { onSubmitted: (runId: string) => void }) {
  const [ablation, setAblation] = useState<string | ''>('')
  const [overrides, setOverrides] = useState<any>({})
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatch] = useState(8)
  const [maxLen, setMaxLen] = useState(128)
  const [saveDir, setSaveDir] = useState<string>('')
  const [saveArtifacts, setSaveArtifacts] = useState(false)
  const [earlyStop, setEarlyStop] = useState<string>('0.0') // epoch1_val_below
  const [gradChkpt, setGradChkpt] = useState(true)
  const [device, setDevice] = useState<'auto'|'cuda'|'cpu'>('auto')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Use a new storage key to avoid older defaults (where saveArtifacts was true)
    const saved = localStorage.getItem('form_state_v2')
    if (saved) {
      const s = JSON.parse(saved)
      setAblation(s.ablation ?? '')
      setOverrides(s.overrides ?? {})
      setEpochs(s.epochs ?? 3)
      setBatch(s.batchSize ?? 8)
      setMaxLen(s.maxLen ?? 128)
      setSaveDir(s.saveDir ?? '')
      // Default remains unchecked unless explicitly saved as true in v2
      setSaveArtifacts(Boolean(s.saveArtifacts))
      setEarlyStop(String(s.earlyStop ?? '0.0'))
      setGradChkpt(s.gradChkpt ?? true)
      setDevice(s.device ?? 'auto')
    }
  }, [])

  const payload = useMemo(() => ({
    ablation: ablation || undefined,
    overrides,
    epochs,
    batch_size: batchSize,
    max_len: maxLen,
    save_dir: saveDir || undefined,            // disabled by default if empty (your backend treats None as disabled)
    save_artifacts: saveArtifacts,
    early_stop: earlyStop ? { epoch1_val_below: Number(earlyStop) } : undefined,
    gradient_checkpointing: gradChkpt,
    device: device === 'auto' ? undefined : device
  }), [ablation, overrides, epochs, batchSize, maxLen, saveDir, saveArtifacts, earlyStop, gradChkpt, device])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      localStorage.setItem('form_state_v2', JSON.stringify({
        ablation, overrides, epochs, batchSize, maxLen, saveDir, saveArtifacts, earlyStop, gradChkpt, device
      }))
      const res = await createRun(payload)
      onSubmitted(res.run_id)
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Submit failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 p-4 rounded-lg bg-zinc-900 border border-zinc-800">
          <label className="block text-sm font-semibold mb-2">Ablation Preset (optional)</label>
          <div className="flex gap-2">
            <select
              className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2"
              value={ablation}
              onChange={e => setAblation(e.target.value)}
            >
              <option value="">(custom)</option>
              {ABLATIONS.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <button
              type="button"
              className="px-2 text-xs bg-zinc-800 rounded-md"
              onClick={() => setAblation('')}
              title="Clear"
            >Clear</button>
          </div>

          <div className="mt-4">
            <label className="block text-sm font-semibold mb-2">Overrides (JSON)</label>
            <JSONEditor value={overrides} onChange={setOverrides}/>
            <p className="text-xs text-zinc-400 mt-1">
              Any training or BridgeCfg knob, e.g. <code>{"{\"hdim_proj_dim\":32, \"route_pool\":\"cls\", \"lr_encoder\":1.5e-5}"}</code>
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
            <label className="block text-sm font-semibold">Epochs</label>
            <input className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" type="number" min={1} value={epochs} onChange={e=>setEpochs(+e.target.value)} />
            <label className="block text-sm font-semibold mt-3">Batch size</label>
            <input className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" type="number" min={1} value={batchSize} onChange={e=>setBatch(+e.target.value)} />
            <label className="block text-sm font-semibold mt-3">Max length</label>
            <input className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" type="number" min={16} value={maxLen} onChange={e=>setMaxLen(+e.target.value)} />
          </div>

          <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
            <label className="block text-sm font-semibold">Save dir (optional)</label>
            <input className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" placeholder="runs/exp1" value={saveDir} onChange={e=>setSaveDir(e.target.value)} />
            <div className="mt-2 flex items-center gap-2">
              <input id="art" type="checkbox" className="rounded" checked={saveArtifacts} onChange={e=>setSaveArtifacts(e.target.checked)} />
              <label htmlFor="art" className="text-sm">Save artifacts (model.pt)</label>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
            <label className="block text-sm font-semibold">Early stop if epoch1 &lt;=</label>
            <input className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" type="number" step="0.01" value={earlyStop} onChange={e=>setEarlyStop(e.target.value)} />
            <div className="mt-2 flex items-center gap-2">
              <input id="gc" type="checkbox" className="rounded" checked={gradChkpt} onChange={e=>setGradChkpt(e.target.checked)} />
              <label htmlFor="gc" className="text-sm">Gradient checkpointing</label>
            </div>
            <label className="block text-sm font-semibold mt-3">Device</label>
            <select className="w-full bg-zinc-950 border border-zinc-800 rounded-md p-2 mt-1" value={device} onChange={e=>setDevice(e.target.value as any)}>
              <option value="auto">(auto)</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>

          <button
            disabled={submitting}
            className="w-full py-2 rounded-md bg-emerald-600 hover:bg-emerald-700 font-semibold disabled:opacity-60"
          >
            {submitting ? 'Submitting…' : 'Run Ablation'}
          </button>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </div>
      </div>
    </form>
  )
}
