import type { RunDetail as RunDetailT } from '../lib/types'
import MetricCard from './MetricCard'
import ChartLine from './ChartLine'
import EpochsTable from './EpochsTable'

export default function RunDetail({ data }: { data: RunDetailT }) {
  const cfg = data.result?.cfg
  const epochs = data.result?.epochs || []
  const nsys = data.result?.profiling?.nsys_summary
  const nsysDetails = data.result?.profiling?.nsys_details
  const download = (obj: any, name: string) => {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-4 gap-3">
        <MetricCard label="Status" value={data.status} />
        <MetricCard label="Best val_acc" value={data.result?.best?.val_acc?.toFixed?.(4) ?? '—'}
          sub={data.result?.best ? `epoch ${data.result.best.epoch}` : ''} />
        <MetricCard label="Device" value={data.device ?? 'auto'} />
        <MetricCard label="Ablation" value={data.ablation ?? '(custom)'} />
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <ChartLine
          data={(data.epochs_rows || []).map(e => ({ epoch: e.epoch, val_acc: e.val_acc, train_acc: e.train_acc }))}
          xKey="epoch" yKey="val_acc" y2Key="train_acc" yLabel="accuracy"
        />
        <ChartLine
          data={epochs.map(e => ({ epoch: e.epoch, f1_macro: e.val_f1_macro }))}
          xKey="epoch" yKey="f1_macro" yLabel="F1 (macro)"
        />
      </div>

      {nsys && (
        <div>
          <div className="text-sm font-semibold mb-2">Profiling (Nsight Systems)</div>
          <div className="grid md:grid-cols-5 gap-3">
            <MetricCard label="GPU Busy %" value={
              typeof nsys.gpu_busy_pct === 'number' ? nsys.gpu_busy_pct.toFixed(1) + '%' : '—'
            } />
            <MetricCard label="Transfer Overhead" value={
              typeof nsys.transfer_overhead_pct === 'number' ? nsys.transfer_overhead_pct.toFixed(1) + '%' : '—'
            } />
            <MetricCard label="GPU Time" value={
              typeof nsys.total_gpu_time_ms === 'number' ? (nsys.total_gpu_time_ms/1000).toFixed(2) + 's' : '—'
            } />
            <MetricCard label="Memcpy Time" value={
              typeof nsys.memcpy_time_ms === 'number' ? (nsys.memcpy_time_ms/1000).toFixed(2) + 's' : '—'
            } />
            <MetricCard label="# Kernels" value={
              typeof nsys.num_unique_kernels === 'number' ? nsys.num_unique_kernels : '—'
            } />
          </div>
        </div>
      )}

      {!nsys && (data.status === 'RUNNING' || data.status === 'STARTED') && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-1">Profiling (Nsight Systems)</div>
          <div className="text-xs text-zinc-400">
            Profiling metrics will appear after the run completes. The timeline and CSV stats are exported at the end.
          </div>
        </div>
      )}

      {!nsys && data.status === 'COMPLETE' && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-1">Profiling (Nsight Systems)</div>
          <div className="text-xs text-zinc-400">
            No parsed metrics available. The timeline file is saved in Artifacts; open it in Nsight Systems to explore.
          </div>
        </div>
      )}

      {nsysDetails && (
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 overflow-auto">
            <div className="text-sm font-semibold mb-2">Top GPU Kernels</div>
            <table className="w-full text-xs">
              <thead className="text-zinc-400">
                <tr>
                  <th className="text-left">Kernel</th>
                  <th className="text-right">Calls</th>
                  <th className="text-right">Time (ms)</th>
                  <th className="text-right">Avg (ms)</th>
                  <th className="text-right">% GPU</th>
                </tr>
              </thead>
              <tbody>
                {(nsysDetails.kernels || []).slice(0, 15).map((k, i) => (
                  <tr key={i} className="border-t border-zinc-800">
                    <td className="pr-2 truncate max-w-[22rem]" title={k.name}>{k.name}</td>
                    <td className="text-right">{k.calls ?? '—'}</td>
                    <td className="text-right">{typeof k.time_ms==='number'?k.time_ms.toFixed(2):'—'}</td>
                    <td className="text-right">{typeof k.avg_ms==='number'?k.avg_ms.toFixed(3):'—'}</td>
                    <td className="text-right">{typeof k.pct_gpu_time==='number'?k.pct_gpu_time.toFixed(1)+'%':'—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="space-y-4">
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 overflow-auto">
              <div className="text-sm font-semibold mb-2">CUDA API Summary</div>
              <table className="w-full text-xs">
                <thead className="text-zinc-400">
                  <tr>
                    <th className="text-left">API</th>
                    <th className="text-right">Calls</th>
                    <th className="text-right">Time (ms)</th>
                    <th className="text-right">Avg (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {(nsysDetails.cuda_api || []).slice(0, 15).map((r, i) => (
                    <tr key={i} className="border-t border-zinc-800">
                      <td className="pr-2 truncate max-w-[22rem]" title={r.name}>{r.name}</td>
                      <td className="text-right">{r.calls ?? '—'}</td>
                      <td className="text-right">{typeof r.time_ms==='number'?r.time_ms.toFixed(2):'—'}</td>
                      <td className="text-right">{typeof r.avg_ms==='number'?r.avg_ms.toFixed(3):'—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 overflow-auto">
              <div className="text-sm font-semibold mb-2">GPU Transfers</div>
              <table className="w-full text-xs">
                <thead className="text-zinc-400">
                  <tr>
                    <th className="text-left">Op</th>
                    <th className="text-right">Calls</th>
                    <th className="text-right">Time (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {(nsysDetails.transfers || []).slice(0, 15).map((r, i) => (
                    <tr key={i} className="border-t border-zinc-800">
                      <td className="pr-2 truncate max-w-[22rem]" title={r.name}>{r.name}</td>
                      <td className="text-right">{r.calls ?? '—'}</td>
                      <td className="text-right">{typeof r.time_ms==='number'?r.time_ms.toFixed(2):'—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-semibold">Overrides</div>
            <button className="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" onClick={() => download(data.overrides ?? {}, `overrides-${data.id}.json`)}>Download</button>
          </div>
          <pre className="text-xs overflow-auto bg-zinc-950 p-3 rounded border border-zinc-800">
            {JSON.stringify(data.overrides ?? {}, null, 2)}
          </pre>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-semibold">BridgeCfg (effective)</div>
            <button className="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" onClick={() => download(cfg ?? {}, `cfg-${data.id}.json`)}>Download</button>
          </div>
          <pre className="text-xs overflow-auto bg-zinc-950 p-3 rounded border border-zinc-800">
            {JSON.stringify(cfg ?? {}, null, 2)}
          </pre>
        </div>
      </div>

      <div>
        <div className="text-sm font-semibold mb-2">Epoch Breakdown</div>
        <EpochsTable data={data} />
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <div className="text-sm font-semibold mb-2">Artifacts</div>
        {data.artifacts && data.artifacts.length > 0 ? (
          <ul className="text-sm list-disc pl-6">
            {data.artifacts.map(a => (
              <li key={a.id}>
                <span className="text-zinc-400 mr-2">[{a.kind || 'file'}]</span>
                <span className="font-mono">{a.path}</span>
                {typeof a.size === 'number' && <span className="text-zinc-400 ml-2">({(a.size/1024).toFixed(1)} KB)</span>}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm text-zinc-400">No artifacts recorded.</div>
        )}
      </div>

      {data.error && (
        <div className="bg-red-950 border border-red-800 rounded-lg p-4">
          <div className="text-sm font-semibold mb-2">Error</div>
          <pre className="text-xs overflow-auto">{data.error}</pre>
        </div>
      )}
    </div>
  )
}
