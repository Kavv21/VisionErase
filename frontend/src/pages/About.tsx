const STATS = [
  { value: 'Built for creators', desc: 'From solo filmmakers to production teams' },
  { value: '4K support', desc: 'Process full-resolution footage without quality loss' },
  { value: 'Parallel processing', desc: 'Multi-segment pipeline for long-form content' },
]

export default function About() {
  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-16">
        <p className="text-sm font-semibold text-red-600 mb-6">• About us</p>
        <h1 className="text-6xl md:text-7xl font-black text-stone-900 leading-[0.95] tracking-tight max-w-3xl mb-8">
          Remove anything. Every frame. Automatically.
        </h1>
        <p className="text-lg text-stone-500 max-w-2xl leading-relaxed">
          VisionErase was built around a single insight: removing objects from videos is
          dramatically harder than from images — yet people need it constantly.
        </p>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100">
            <h2 className="text-2xl font-black text-stone-900 mb-4">The problem</h2>
            <p className="text-stone-500 leading-relaxed">
              Video object removal requires tracking across hundreds or thousands of frames,
              handling occlusions where the object disappears behind something else, and
              maintaining temporal consistency so the inpainting doesn't flicker.
              Professional tools cost tens of thousands of dollars and still require
              manual frame-by-frame cleanup.
            </p>
          </div>
          <div className="bg-white rounded-2xl p-8 shadow-sm border border-stone-100">
            <h2 className="text-2xl font-black text-stone-900 mb-4">Our solution</h2>
            <p className="text-stone-500 leading-relaxed">
              We chain SAM 2 for one-shot segmentation, XMem++ for occlusion-robust
              tracking, and ProPainter for inpainting — then run them in parallel across
              video segments and stitch with our proprietary BoundaryFusion model that
              eliminates temporal seams at segment boundaries.
            </p>
          </div>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-24">
        <p className="text-sm font-semibold text-red-600 mb-3">• Capability</p>
        <h2 className="text-4xl font-black text-stone-900 tracking-tight mb-10">
          Built for scale
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {STATS.map((s) => (
            <div
              key={s.value}
              className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100"
            >
              <div className="text-xl font-black text-stone-900 mb-1">{s.value}</div>
              <div className="text-sm text-stone-500">{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="max-w-6xl mx-auto px-6 py-10 flex items-center justify-between text-sm text-stone-400 border-t border-stone-200">
        <span className="font-bold text-stone-900">VisionErase</span>
        <span>© 2026 VisionErase. All rights reserved.</span>
      </footer>
    </div>
  )
}
