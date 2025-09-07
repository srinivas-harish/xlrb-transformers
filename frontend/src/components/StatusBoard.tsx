import { useEffect, useMemo, useState } from 'react'
import { listRuns } from '../lib/api'
import type { RunRow } from '../lib/types'
import { relTime } from '../lib/fmt'

type Group = 'QUEUED'|'RUNNING'|'COMPLETE'|'FAILED'

export default function StatusBoard({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<RunRow[]>([])

  useEffect(() => {
    let tm: any
    const poll = async () => {
      try { setRows(await listRuns(undefined, 200)) } catch {}
      tm = setTimeout(poll, 1500)
    }
    poll()
    return () => tm && clearTimeout(tm)
  }, [])

  const grouped = useMemo(() => {
    const g: Record<Group, RunRow[]> = { QUEUED: [], RUNNING: [], COMPLETE: [], FAILED: [] }
    rows.forEach(r => {
      const key: Group = (r.status === 'STARTED') ? 'RUNNING' : (r.status as Group)
      ;(g[key] ||= []).push(r)
    })
    return g
  }, [rows])

  return (
    <div className="grid md:grid-cols-4 gap-4">
      {(['QUEUED','RUNNING','COMPLETE','FAILED'] as Group[]).map(gr => (
        <div key={gr} className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-semibold">{gr}</div>
            <div className="text-xs text-zinc-400">{grouped[gr].length}</div>
          </div>
          <div className="space-y-2 overflow-auto">
            {grouped[gr].map(r => (
              <button key={r.id}
                className="w-full text-left p-3 bg-zinc-950/50 border border-zinc-800 rounded-md hover:bg-zinc-800/60"
                onClick={() => onOpen(r.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="font-mono text-xs">{r.id.slice(0,8)}…</div>
                  <div className="text-xs text-zinc-400">{relTime(r.created_at)}</div>
                </div>
                <div className="mt-1 text-sm">
                  {r.ablation || '(custom)'}
                </div>
                <div className="mt-1 text-xs text-zinc-400 flex gap-3">
                  <span>epochs {r.epochs}</span>
                  <span>batch {r.batch_size}</span>
                  {typeof r.best_val_acc === 'number' && (
                    <span>best {r.best_val_acc.toFixed(4)}</span>
                  )}
                </div>
              </button>
            ))}
            {grouped[gr].length === 0 && (
              <div className="text-sm text-zinc-500 p-3">No runs.</div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
