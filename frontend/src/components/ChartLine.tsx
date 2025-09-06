import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function ChartLine({
  data, xKey, yKey, y2Key, height = 220, yLabel
}: { data: any[]; xKey: string; yKey: string; y2Key?: string; height?: number; yLabel?: string }) {
  return (
    <div className="w-full h-[220px] bg-zinc-900 border border-zinc-800 rounded-lg p-3">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey={xKey} stroke="#a1a1aa" />
          <YAxis stroke="#a1a1aa" label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#a1a1aa' } : undefined} />
          <Tooltip contentStyle={{ background: '#09090b', border: '1px solid #27272a' }} />
          <Line type="monotone" dataKey={yKey} dot={false} />
          {y2Key && <Line type="monotone" dataKey={y2Key} dot={false} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
