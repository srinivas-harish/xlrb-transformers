import { useEffect, useMemo, useState } from 'react'
import { getRun } from '../lib/api'
import type { RunDetail as RunDetailT } from '../lib/types'
import MetricCard from './MetricCard'
import ChartLine from './ChartLine'
import { pct } from '../lib/fmt'
import { clsx } from 'clsx'

export default function RunProgress({ runId, run }: { runId: string; run?: RunDetailT }) {
  const [data, setData] = useState<RunDetailT | undefined>(run)

  useEffect(() => { if (run) setData(run) }, [run])
  useEffect(() => {
    let tm: any
    const poll = async () => {
      try { setData(await getRun(runId)) } catch {}
      tm = setTimeout(poll, data?.status === 'COMPLETE' || data?.status === 'FAILED' ? 1500 : 800)
    }
    poll()
    return () => tm && clearTimeout(tm)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  const epochsRequested = data?.epochs ?? 0
  const epochsDone = data?.epochs_rows?.length ?? 0
  const progress = epochsRequested ? epochsDone / epochsRequested : 0
  const last = data?.epochs_rows?.[data.epochs_rows.length - 1]
  const lastVal = last?.val_acc ?? data?.result?.best?.val_acc

  const accData = useMemo(() => (data?.epochs_rows || []).map(e => ({
    epoch: e.epoch, val_acc: e.val_acc, train_acc: e.train_acc
  })), [data])

  return (
    <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs text-zinc-400">Run ID</div>
          <div className="font-mono text-sm">{runId}</div>
        </div>
        <div className="flex items-center gap-2">
          <span className={clsx(
            'px-2 py-1 rounded text-xs font-bold',
            data?.status === 'COMPLETE' ? 'bg-emerald-700' :
            data?.status === 'FAILED' ? 'bg-red-700' :
            data?.status === 'STARTED' ? 'bg-amber-700' : 'bg-zinc-700'
          )}>
            {data?.status ?? '…'}
          </span>
        </div>
      </div>

      <div className="mt-4">
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-emerald-600" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        <div className="flex justify-between mt-1 text-xs text-zinc-400">
          <div>{epochsDone} / {epochsRequested} epochs</div>
          <div>val_acc: {lastVal?.toFixed?.(4) ?? '—'}</div>
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-3 mt-4">
        <MetricCard label="Best val_acc" value={data?.result?.best?.val_acc?.toFixed?.(4) ?? '—'}
          sub={data?.result?.best ? `at epoch ${data.result.best.epoch}` : ''} />
        <MetricCard label="Batch size" value={data?.batch_size ?? '—'} />
        <MetricCard label="Max length" value={data?.max_len ?? '—'} />
        <MetricCard label="Device" value={data?.device ?? 'auto'} />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <ChartLine data={accData} xKey="epoch" yKey="val_acc" y2Key="train_acc" yLabel="accuracy" />
        <ChartLine
          data={(data?.result?.epochs || []).map((e) => ({
            epoch: e.epoch, alpha_hdim_mean: e.gates?.alpha_hdim?.mean
          }))}
          xKey="epoch" yKey="alpha_hdim_mean" yLabel="α_hdim mean"
        />
      </div>
    </div>
  )
}
