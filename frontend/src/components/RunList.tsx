import { useEffect, useMemo, useState } from 'react'
import { listRuns } from '../lib/api'
import type { RunRow } from '../lib/types'
import { relTime } from '../lib/fmt'
import { clsx } from 'clsx'

export default function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<RunRow[]>([])
  const [statusFilter, setStatusFilter] = useState<string>('all')
  useEffect(() => {
    let tm: any
    const poll = async () => {
      try { setRows(await listRuns(undefined, 100)) } catch {}
      tm = setTimeout(poll, 1500)
    }
    poll()
    return () => tm && clearTimeout(tm)
  }, [])

  const counts = useMemo(() => {
    const c = { all: rows.length, QUEUED: 0, STARTED: 0, COMPLETE: 0, FAILED: 0 } as Record<string, number>
    rows.forEach(r => { c[r.status] = (c[r.status] || 0) + 1 })
    return c
  }, [rows])

  const filtered = useMemo(() => rows.filter(r => statusFilter === 'all' ? true : r.status === statusFilter), [rows, statusFilter])

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 bg-zinc-900 border-b border-zinc-800">
        <div className="text-sm font-semibold">Runs</div>
        <div className="flex gap-2">
          {(['all','QUEUED','STARTED','COMPLETE','FAILED'] as const).map(s => (
            <button key={s}
              className={clsx('px-2 py-1 rounded text-xs', statusFilter===s ? 'bg-zinc-800' : 'bg-zinc-950 hover:bg-zinc-800')}
              onClick={() => setStatusFilter(s)}
            >{s} <span className="text-zinc-400">{counts[s]}</span></button>
          ))}
        </div>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-zinc-800 text-zinc-300">
          <tr>
            <th className="text-left px-3 py-2">ID</th>
            <th className="text-left px-3 py-2">Status</th>
            <th className="text-left px-3 py-2">Ablation</th>
            <th className="text-left px-3 py-2">Epochs</th>
            <th className="text-left px-3 py-2">Batch</th>
            <th className="text-left px-3 py-2">Best acc</th>
            <th className="text-left px-3 py-2">Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(r => (
            <tr key={r.id} className="border-t border-zinc-800 hover:bg-zinc-800/50">
              <td className="px-3 py-2 font-mono text-xs">{r.id.slice(0,8)}…</td>
              <td className="px-3 py-2">
                <span className={clsx(
                  'px-2 py-1 rounded text-xs font-bold',
                  r.status === 'COMPLETE' ? 'bg-emerald-700' :
                  r.status === 'FAILED' ? 'bg-red-700' :
                  (r.status === 'RUNNING' || r.status === 'STARTED') ? 'bg-amber-700' : 'bg-zinc-700'
                )}>{r.status}</span>
              </td>
              <td className="px-3 py-2">{r.ablation ?? '—'}</td>
              <td className="px-3 py-2">{r.epochs}</td>
              <td className="px-3 py-2">{r.batch_size}</td>
              <td className="px-3 py-2">{r.best_val_acc?.toFixed?.(4) ?? '—'}</td>
              <td className="px-3 py-2">{relTime(r.created_at)}</td>
              <td className="px-3 py-2 text-right">
                <button className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" onClick={() => onOpen(r.id)}>
                  Open
                </button>
              </td>
            </tr>
          ))}
          {filtered.length === 0 && (
            <tr><td colSpan={8} className="px-3 py-6 text-center text-zinc-400">No runs yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
