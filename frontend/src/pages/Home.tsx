import { useState, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion, useInView, AnimatePresence, animate } from 'framer-motion'
import { useEffect } from 'react'
import GlassCard from '../components/GlassCard'
import HeroAnimation from '../components/HeroAnimation'

const FEATURES = [
  {
    title: 'SAM2 Segmentation',
    desc: 'Click any object, AI masks it instantly.',
    icon: (
      <svg className="w-6 h-6 text-[#A78BFA]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" />
      </svg>
    ),
  },
  {
    title: 'XMem++ Tracking',
    desc: 'Mask follows the object across every frame.',
    icon: (
      <svg className="w-6 h-6 text-[#A78BFA]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    title: 'ProPainter Removal',
    desc: 'Object erased, background filled naturally.',
    icon: (
      <svg className="w-6 h-6 text-[#A78BFA]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
      </svg>
    ),
  },
]

const STEPS = [
  { n: '01', title: 'Upload your video' },
  { n: '02', title: 'Click the object to remove' },
  { n: '03', title: 'AI tracks it across frames' },
  { n: '04', title: 'Object is erased naturally' },
  { n: '05', title: 'Download your clean video' },
]

const STATS: { label: string; value: number; suffix: string; prefix: string }[] = [
  { label: 'Per frame', value: 10, suffix: 's', prefix: '< ' },
  { label: 'Mask accuracy', value: 99, suffix: '%', prefix: '' },
  { label: 'Resolution support', value: 4, suffix: 'K', prefix: '' },
]

function CountUpStat({ stat }: { stat: (typeof STATS)[number] }) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    if (!inView) return
    const controls = animate(0, stat.value, {
      duration: 1.2,
      ease: 'easeOut',
      onUpdate: (v) => setDisplay(Math.round(v)),
    })
    return () => controls.stop()
  }, [inView, stat.value])

  return (
    <div ref={ref} className="text-center">
      <div className="text-4xl md:text-5xl font-black bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent mb-1">
        {stat.prefix}{display}{stat.suffix}
      </div>
      <div className="text-sm text-[#A0A0B0]">{stat.label}</div>
    </div>
  )
}

export default function Home() {
  const [demoOpen, setDemoOpen] = useState(false)
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#0A0A0F] overflow-hidden">
      {/* Hero */}
      <section className="relative min-h-screen overflow-hidden bg-black">
        <HeroAnimation />

        {/* Hero text */}
        <div className="relative z-10 flex flex-col items-center justify-center min-h-screen text-center px-6 -translate-y-16">
          <p className="text-white/30 text-xs tracking-[0.3em] uppercase mb-8 font-light">
            Powered by SAM2 · XMem++ · ProPainter
          </p>

          <h1
            className="text-white leading-[1.05] tracking-tight mb-8"
            style={{ fontFamily: "'Instrument Serif', serif", fontSize: 'clamp(3.5rem, 8vw, 9rem)' }}
          >
            Remove Anything<br />
            <em className="italic text-white/50">From Any Video.</em>
          </h1>

          <div className="flex gap-4 mt-4">
            <button
              onClick={() => navigate('/upload')}
              className="bg-white text-black rounded-full px-8 py-3 text-sm font-medium hover:bg-white/90 transition"
            >
              Upload Video →
            </button>
            <button
              onClick={() => setDemoOpen(true)}
              className="liquid-glass rounded-full px-8 py-3 text-white text-sm font-medium hover:bg-white/5 transition"
            >
              Watch it work
            </button>
          </div>
        </div>
      </section>

      {/* Demo modal */}
      <AnimatePresence>
        {demoOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/70 backdrop-blur-sm flex items-center justify-center px-4"
            onClick={() => setDemoOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-3xl aspect-video backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center relative"
            >
              <button
                onClick={() => setDemoOpen(false)}
                className="absolute top-4 right-4 text-[#A0A0B0] hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#7C3AED] to-[#2563EB] flex items-center justify-center shadow-[0_0_40px_rgba(124,58,237,0.5)]">
                <svg className="w-8 h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Feature cards */}
      <section className="max-w-6xl mx-auto px-6 py-24">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {FEATURES.map((f, i) => (
            <GlassCard
              key={f.title}
              hover
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="p-7 border-t-2 border-t-[#7C3AED] relative"
            >
              <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-5">
                {f.icon}
              </div>
              <h3 className="font-bold text-[#F8F8FF] mb-2">{f.title}</h3>
              <p className="text-sm text-[#A0A0B0] leading-relaxed">{f.desc}</p>
            </GlassCard>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-4xl mx-auto px-6 py-24">
        <h2 className="text-3xl md:text-4xl font-black text-center text-[#F8F8FF] mb-16">
          How it works
        </h2>
        <div className="relative">
          <div className="absolute left-[27px] top-2 bottom-2 w-px border-l-2 border-dotted border-[#7C3AED]/40" />
          <div className="space-y-8">
            {STEPS.map((step, i) => (
              <motion.div
                key={step.n}
                initial={{ opacity: 0, x: -16 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="relative flex items-center gap-6"
              >
                <span className="relative z-10 w-14 h-14 shrink-0 rounded-full bg-[#12121A] border border-[#7C3AED]/50 flex items-center justify-center font-mono font-bold text-[#A78BFA]">
                  {step.n}
                </span>
                <span className="text-[#F8F8FF] font-medium">{step.title}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats strip */}
      <section className="max-w-4xl mx-auto px-6 py-16">
        <GlassCard className="grid grid-cols-1 md:grid-cols-3 gap-8 p-10">
          {STATS.map((s) => (
            <CountUpStat key={s.label} stat={s} />
          ))}
        </GlassCard>
      </section>

      {/* Footer */}
      <footer className="max-w-6xl mx-auto px-6 py-10 flex items-center justify-between text-sm text-[#A0A0B0] border-t border-white/10">
        <span className="font-black bg-gradient-to-r from-[#A78BFA] to-[#60A5FA] bg-clip-text text-transparent">
          VisionErase
        </span>
        <div className="flex items-center gap-6">
          <Link to="/about" className="hover:text-white transition-colors">About</Link>
          <Link to="/features" className="hover:text-white transition-colors">Features</Link>
          <span>© 2026 VisionErase</span>
        </div>
      </footer>
    </div>
  )
}
