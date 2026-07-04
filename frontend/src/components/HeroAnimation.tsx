import { useEffect, useRef, useState } from 'react'
import { motion, useAnimationControls } from 'framer-motion'

const PARTICLES = Array.from({ length: 8 }, (_, i) => {
  const angle = (i / 8) * Math.PI * 2
  return { id: i, dx: Math.cos(angle) * 130, dy: Math.sin(angle) * 130 }
})

const TRAIL_DOTS = [
  { id: 0, cx: 560, cy: 420 },
  { id: 1, cx: 545, cy: 425 },
  { id: 2, cx: 530, cy: 430 },
]

function wait(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

export default function HeroAnimation() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [label, setLabel] = useState('Tennis player detected')
  const [frame, setFrame] = useState(1)

  const playerControls = useAnimationControls()
  const maskControls = useAnimationControls()
  const labelControls = useAnimationControls()
  const glowControls = useAnimationControls()
  const particleControls = useAnimationControls()
  const trailControls = useAnimationControls()
  const frameCounterControls = useAnimationControls()

  // Layer 1: background video crossfade loop
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let rafId = 0
    let resetTimeout = 0

    const fadeTo = (target: number, duration: number) => {
      cancelAnimationFrame(rafId)
      const start = performance.now()
      const from = parseFloat(video.style.opacity || '0')
      const step = (now: number) => {
        const t = Math.min((now - start) / duration, 1)
        video.style.opacity = String(from + (target - from) * t)
        if (t < 1) rafId = requestAnimationFrame(step)
      }
      rafId = requestAnimationFrame(step)
    }

    const handleCanPlay = () => {
      video.play().catch(() => {})
      fadeTo(1, 500)
    }
    const handleTimeUpdate = () => {
      if (video.duration && video.currentTime >= video.duration - 0.55) {
        fadeTo(0, 500)
      }
    }
    const handleEnded = () => {
      video.style.opacity = '0'
      resetTimeout = window.setTimeout(() => {
        video.currentTime = 0
        video.play().catch(() => {})
        fadeTo(1, 500)
      }, 100)
    }

    video.addEventListener('canplay', handleCanPlay)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('ended', handleEnded)

    return () => {
      video.removeEventListener('canplay', handleCanPlay)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('ended', handleEnded)
      cancelAnimationFrame(rafId)
      window.clearTimeout(resetTimeout)
    }
  }, [])

  // Layer 3: SVG tennis animation sequence
  useEffect(() => {
    let active = true
    let frameInterval = 0

    const runSequence = async () => {
      while (active) {
        // reset
        setLabel('Tennis player detected')
        setFrame(1)
        playerControls.set({ opacity: 0, x: 0, filter: 'blur(0px)' })
        maskControls.set({ opacity: 1, strokeDashoffset: 440, fillOpacity: 0.12, x: 0 })
        labelControls.set({ opacity: 0, borderColor: 'rgba(255,255,255,0.15)' })
        glowControls.set({ opacity: 0 })
        particleControls.set({ opacity: 0, scale: 0, x: 0, y: 0 })
        trailControls.set({ opacity: 0 })
        frameCounterControls.set({ opacity: 1 })

        if (!active) break

        // Phase 1 (0-1.5s): player appears
        labelControls.start({ opacity: 1 }, { duration: 0.4 })
        await playerControls.start({ opacity: 1 }, { duration: 1 })
        if (!active) break
        await wait(500)
        if (!active) break

        // Phase 2 (1.5-3.5s): mask draws
        setLabel('Segmenting object...')
        await Promise.all([
          maskControls.start({ strokeDashoffset: 0 }, { duration: 2, ease: 'easeInOut' }),
          maskControls.start({ fillOpacity: [0.12, 0.24, 0.12] }, { duration: 1, repeat: 1, ease: 'easeInOut' }),
          glowControls.start({ opacity: 1 }, { duration: 0.6, delay: 0.6 }),
        ])
        if (!active) break

        // Phase 3 (3.5-5.5s): tracking
        setLabel('Tracking across frames...')
        frameInterval = window.setInterval(() => {
          setFrame((f) => (f >= 60 ? 1 : f + 1))
        }, 200)
        trailControls.start({ opacity: [0, 0.5, 0] }, { duration: 2, ease: 'easeInOut' })
        await Promise.all([
          playerControls.start({ x: [0, -8, 8, 0] }, { duration: 2, ease: 'easeInOut' }),
          maskControls.start({ x: [0, -8, 8, 0] }, { duration: 2, ease: 'easeInOut' }),
        ])
        window.clearInterval(frameInterval)
        if (!active) break

        // Phase 4 (5.5-7.5s): removal
        setLabel('Removing with ProPainter...')
        particleControls.start(
          (custom: { dx: number; dy: number }) => ({
            opacity: [0, 1, 0],
            scale: [0, 2, 0],
            x: [0, custom.dx, custom.dx * 1.3],
            y: [0, custom.dy, custom.dy * 1.3],
          }),
          { duration: 1.8, ease: 'easeOut' },
        )
        await Promise.all([
          playerControls.start({ opacity: 0, filter: 'blur(8px)' }, { duration: 1.8 }),
          maskControls.start({ fillOpacity: [0.12, 0.3, 0.12, 0.3, 0.12] }, { duration: 1.6, ease: 'easeInOut' }),
        ])
        if (!active) break

        // Phase 5 (7.5-9s): clean result
        setLabel('✓ Object removed')
        labelControls.start({ borderColor: 'rgba(74,222,128,0.5)' }, { duration: 0.4 })
        glowControls.start({ opacity: 0 }, { duration: 0.8 })
        await maskControls.start({ opacity: 0 }, { duration: 1 })
        if (!active) break
        await wait(500)
        if (!active) break

        // Phase 6 (9-9.5s): fade all
        await Promise.all([
          labelControls.start({ opacity: 0 }, { duration: 0.5 }),
          frameCounterControls.start({ opacity: 0 }, { duration: 0.5 }),
        ])
      }
    }

    runSequence()

    return () => {
      active = false
      window.clearInterval(frameInterval)
      playerControls.stop()
      maskControls.stop()
      labelControls.stop()
      glowControls.stop()
      particleControls.stop()
      trailControls.stop()
      frameCounterControls.stop()
    }
  }, [playerControls, maskControls, labelControls, glowControls, particleControls, trailControls, frameCounterControls])

  return (
    <>
      <video
        ref={videoRef}
        src="/hero.mp4"
        muted
        playsInline
        preload="auto"
        className="absolute inset-0 w-full h-full object-cover"
        style={{ opacity: 0 }}
      />

      <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-black/20 to-black/70" />

      <svg
        viewBox="0 0 1200 675"
        preserveAspectRatio="xMidYMid slice"
        className="absolute inset-0 w-full h-full"
      >
        {TRAIL_DOTS.map((dot) => (
          <motion.circle
            key={dot.id}
            cx={dot.cx}
            cy={dot.cy}
            r={3}
            fill="#A78BFA"
            initial={{ opacity: 0 }}
            animate={trailControls}
          />
        ))}

        <motion.g initial={{ opacity: 0, x: 0, filter: 'blur(0px)' }} animate={playerControls}>
          <circle cx={600} cy={320} r={28} stroke="white" strokeWidth={3} fill="none" />
          <rect x={585} y={348} width={30} height={85} stroke="white" strokeWidth={3} fill="none" />
          <line x1={585} y1={370} x2={540} y2={290} stroke="white" strokeWidth={3} />
          <line x1={615} y1={370} x2={665} y2={330} stroke="white" strokeWidth={3} />
          <line x1={590} y1={433} x2={570} y2={510} stroke="white" strokeWidth={3} />
          <line x1={610} y1={433} x2={630} y2={510} stroke="white" strokeWidth={3} />
        </motion.g>

        <motion.circle
          cx={600}
          cy={300}
          r={5}
          fill="#A78BFA"
          initial={{ opacity: 0 }}
          animate={glowControls}
          style={{ filter: 'blur(1px)' }}
        />

        <motion.ellipse
          cx={600}
          cy={415}
          rx={70}
          ry={110}
          stroke="#7C3AED"
          strokeWidth={2.5}
          fill="#7C3AED"
          strokeDasharray={440}
          initial={{ opacity: 1, strokeDashoffset: 440, fillOpacity: 0.12, x: 0 }}
          animate={maskControls}
        />

        {PARTICLES.map((p) => (
          <motion.circle
            key={p.id}
            cx={600}
            cy={415}
            r={4}
            fill="#7C3AED"
            custom={p}
            initial={{ opacity: 0, scale: 0, x: 0, y: 0 }}
            animate={particleControls}
          />
        ))}

        <foreignObject x={470} y={260} width={260} height={40}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={labelControls}
            style={{ borderWidth: 1, borderStyle: 'solid', whiteSpace: 'nowrap' }}
            className="liquid-glass rounded-full px-4 py-2 text-white text-sm text-center"
          >
            {label}
          </motion.div>
        </foreignObject>

        <motion.text
          x={690}
          y={300}
          initial={{ opacity: 1 }}
          animate={frameCounterControls}
          className="fill-white/40 font-mono text-xs"
        >
          {`Frame ${String(frame).padStart(2, '0')} / 60`}
        </motion.text>
      </svg>
    </>
  )
}
