export default function MetricCard({
  label, value, sub
}: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
      <div className="text-xs uppercase tracking-wider text-zinc-400">{label}</div>
      <div className="text-2xl font-extrabold mt-1">{value}</div>
      {sub && <div className="text-xs text-zinc-400 mt-1">{sub}</div>}
    </div>
  )
}
