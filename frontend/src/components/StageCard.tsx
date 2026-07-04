import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

export type StageStatus = 'pending' | 'active' | 'complete' | 'failed'

interface StageCardProps {
  icon: ReactNode
  title: string
  status: StageStatus
  index: number
}

const STATUS_STYLES: Record<StageStatus, string> = {
  pending: 'border-white/10 text-[#A0A0B0]',
  active: 'border-[#7C3AED] shadow-[0_0_24px_rgba(124,58,237,0.25)] text-[#F8F8FF]',
  complete: 'border-[#10B981]/40 text-[#F8F8FF]',
  failed: 'border-[#EF4444]/50 text-[#F8F8FF]',
}

function StatusIndicator({ status }: { status: StageStatus }) {
  if (status === 'complete') {
    return (
      <svg className="w-5 h-5 text-[#10B981]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
      </svg>
    )
  }
  if (status === 'active') {
    return <span className="w-2.5 h-2.5 rounded-full bg-[#7C3AED] animate-pulse" />
  }
  if (status === 'failed') {
    return <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
  }
  return <span className="w-2.5 h-2.5 rounded-full bg-white/15" />
}

export default function StageCard({ icon, title, status, index }: StageCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.06, duration: 0.35 }}
      className={`relative flex items-center gap-4 backdrop-blur-xl bg-white/5 border rounded-2xl px-5 py-4 transition-colors duration-200 overflow-hidden ${STATUS_STYLES[status]}`}
    >
      {status === 'active' && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
          animate={{ x: ['-100%', '200%'] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
        />
      )}
      <div className="relative shrink-0 w-9 h-9 rounded-xl bg-white/5 flex items-center justify-center">
        {icon}
      </div>
      <span className="relative flex-1 font-semibold text-sm">{title}</span>
      <div className="relative shrink-0">
        <StatusIndicator status={status} />
      </div>
    </motion.div>
  )
}
