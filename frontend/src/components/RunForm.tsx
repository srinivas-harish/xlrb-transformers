import { useEffect, useMemo, useState } from 'react'
import JSONEditor from './JSONEditor'
import { createRun, kernelSanity } from '../lib/api'
import type { KernelSanity } from '../lib/types'

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
  const [testing, setTesting] = useState(false)
  const [testRes, setTestRes] = useState<KernelSanity | null>(null)

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

          <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800 mt-4">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-sm font-semibold">GPU Kernel Sanity Test</div>
                <div className="text-xs text-zinc-400">Runs a tiny CUDA matmul under Nsight Compute to confirm kernels are captured.</div>
              </div>
              <button
                type="button"
                disabled={testing}
                onClick={async () => {
                  setTesting(true); setTestRes(null)
                  try {
                    const res = await kernelSanity()
                    setTestRes(res)
                  } catch (e: any) {
                    setTestRes({ ok: false, ncu: { found: false }, python: 'python', kernels: [], num_unique_kernels: 0, stdout: null, stderr: String(e?.message || e), notes: ['Request failed'] })
                  } finally {
                    setTesting(false)
                  }
                }}
                className="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-60"
              >{testing ? 'Testing…' : 'Run Nsight Compute Test'}</button>
            </div>

            {testRes && (
              <div className="space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                    <div><span className="text-zinc-400">NCU found:</span> <span className={testRes.ncu?.found ? 'text-emerald-400' : 'text-red-400'}>{String(!!testRes.ncu?.found)}</span></div>
                    {testRes.ncu?.version && <div><span className="text-zinc-400">NCU version:</span> <span className="font-mono">{testRes.ncu.version}</span></div>}
                    {testRes.ncu?.path && <div className="truncate" title={testRes.ncu.path}><span className="text-zinc-400">NCU path:</span> <span className="font-mono">{testRes.ncu.path}</span></div>}
                    <div><span className="text-zinc-400">WSL:</span> <span className="font-mono">{testRes.env?.is_wsl ? 'yes' : 'no'}</span></div>
                    <div><span className="text-zinc-400">Python:</span> <span className="font-mono">{testRes.python}</span></div>
                    <div><span className="text-zinc-400">torch.cuda.available:</span> <span className={testRes.torch_cuda_available ? 'text-emerald-400' : 'text-red-400'}>{String(!!testRes.torch_cuda_available)}</span></div>
                  </div>
                  <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                    <div><span className="text-zinc-400">Artifact (.ncu-rep):</span> <span className="font-mono">{testRes.artifact || '—'}</span></div>
                    <div><span className="text-zinc-400"># Kernels:</span> <span className="font-mono">{testRes.num_unique_kernels}</span></div>
                    <div><span className="text-zinc-400">Tried sets:</span> <span className="font-mono">{(testRes.tried_sets || []).map(s=>s.join(' ')).join(' | ') || '—'}</span></div>
                    <div><span className="text-zinc-400">OK:</span> <span className={testRes.ok ? 'text-emerald-400' : 'text-red-400'}>{String(testRes.ok)}</span></div>
                  </div>
                </div>

                {testRes.kernels?.length > 0 && (
                  <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                    <div className="text-zinc-300 font-semibold mb-1">Top Kernels</div>
                    <table className="w-full text-xs">
                      <thead className="text-zinc-400">
                        <tr>
                          <th className="text-left">Kernel</th>
                          <th className="text-right">Calls</th>
                          <th className="text-right">Time (ms)</th>
                          <th className="text-right">Avg (ms)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {testRes.kernels.slice(0, 10).map((k, i) => (
                          <tr key={i} className="border-t border-zinc-800">
                            <td className="pr-2 truncate max-w-[16rem]" title={k.name}>{k.name}</td>
                            <td className="text-right">{k.calls ?? '—'}</td>
                            <td className="text-right">{typeof k.time_ms==='number'?k.time_ms.toFixed(2):'—'}</td>
                            <td className="text-right">{typeof k.avg_ms==='number'?k.avg_ms.toFixed(3):'—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {(testRes.raw_preview || testRes.stderr || testRes.stdout) && (
                  <div className="bg-zinc-950 border border-zinc-800 rounded p-2">
                    <div className="text-zinc-300 font-semibold mb-1">Logs</div>
                    {testRes.raw_preview && (
                      <details className="mb-2" open>
                        <summary className="cursor-pointer">NCU Raw CSV (preview)</summary>
                        <pre className="overflow-auto whitespace-pre-wrap text-zinc-300 text-[11px] max-h-56">{testRes.raw_preview}</pre>
                      </details>
                    )}
                    {testRes.stderr && (
                      <details className="mb-2">
                        <summary className="cursor-pointer">stderr</summary>
                        <pre className="overflow-auto whitespace-pre-wrap text-zinc-300 text-[11px] max-h-56">{testRes.stderr}</pre>
                      </details>
                    )}
                    {testRes.stdout && (
                      <details>
                        <summary className="cursor-pointer">stdout</summary>
                        <pre className="overflow-auto whitespace-pre-wrap text-zinc-300 text-[11px] max-h-56">{testRes.stdout}</pre>
                      </details>
                    )}
                    {testRes.notes && testRes.notes.length>0 && (
                      <div className="mt-2 text-zinc-300">
                        <div className="font-semibold mb-1">Notes</div>
                        <ul className="list-disc ml-5">
                          {testRes.notes.map((n,i)=>(<li key={i}>{n}</li>))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </form>
  )
}
