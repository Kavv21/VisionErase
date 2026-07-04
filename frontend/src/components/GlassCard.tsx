import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

interface GlassCardProps extends HTMLMotionProps<'div'> {
  children: ReactNode
  hover?: boolean
  glow?: boolean
}

export default function GlassCard({
  children,
  hover = false,
  glow = false,
  className = '',
  ...props
}: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { scale: 1.02, y: -4 } : undefined}
      transition={{ type: 'spring', stiffness: 300 }}
      className={`backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl ${
        glow ? 'shadow-[0_0_40px_rgba(124,58,237,0.15)]' : ''
      } ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  )
}
