import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

type Variant = 'primary' | 'outline' | 'ghost'

interface GlowButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  children: ReactNode
  variant?: Variant
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary:
    'bg-gradient-to-r from-[#7C3AED] to-[#2563EB] text-white shadow-[0_0_24px_rgba(124,58,237,0.35)] hover:shadow-[0_0_36px_rgba(124,58,237,0.55)]',
  outline:
    'border border-[#7C3AED]/50 text-[#F8F8FF] hover:border-[#7C3AED] hover:bg-[#7C3AED]/10',
  ghost: 'text-[#A0A0B0] hover:text-[#F8F8FF]',
}

export default function GlowButton({
  children,
  variant = 'primary',
  className = '',
  disabled,
  ...props
}: GlowButtonProps) {
  return (
    <motion.button
      whileHover={disabled ? undefined : { scale: 1.02 }}
      whileTap={disabled ? undefined : { scale: 0.97 }}
      transition={{ type: 'spring', stiffness: 300 }}
      disabled={disabled}
      className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ease-out disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none ${VARIANT_CLASSES[variant]} ${className}`}
      {...props}
    >
      {children}
    </motion.button>
  )
}
