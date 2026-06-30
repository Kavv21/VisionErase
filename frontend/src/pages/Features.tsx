const FEATURES = [
  {
    icon: '⬡',
    title: 'SAM 2 segmentation',
    desc: 'One painted frame — SAM 2 produces a pixel-perfect mask for the entire object.',
  },
  {
    icon: '⟳',
    title: 'Occlusion-aware re-identification',
    desc: 'XMem++ tracks objects even when they temporarily disappear behind other subjects.',
  },
  {
    icon: '⧖',
    title: 'Seamless long-form processing',
    desc: 'Videos are split into parallel worker segments and recombined with zero seam artifacts.',
  },
  {
    icon: '✓',
    title: 'Automatic quality checks',
    desc: 'SSIM and PSNR scoring validates every frame before your file is ready for download.',
  },
  {
    icon: '⚡',
    title: 'Fast local inference',
    desc: 'GPU-accelerated pipeline runs models on-device — no external API calls, no data leaving your infrastructure.',
  },
  {
    icon: '⧉',
    title: 'Multiple output formats',
    desc: 'Output is delivered in the same container as your source — MP4, MOV, or WebM.',
  },
]

const CATEGORIES = [
  { n: '001', title: 'Social clips', desc: 'Reels, shorts, and TikToks with removed objects in under 3 minutes.' },
  { n: '002', title: 'Product videos', desc: 'Clean studio-quality footage without manual rotoscoping.' },
  { n: '003', title: 'Interviews & vlogs', desc: 'Remove background distractions while keeping the subject untouched.' },
  { n: '004', title: 'Archival footage', desc: 'Restore and clean up legacy video material at scale.' },
]

export default function Features() {
  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-16">
        <p className="text-sm font-semibold text-red-600 mb-6">• Features</p>
        <h1 className="text-6xl md:text-7xl font-black text-stone-900 leading-[0.95] tracking-tight max-w-3xl mb-6">
          Built on real computer vision research
        </h1>
        <p className="text-lg text-stone-500 max-w-xl">
          Every feature traces back to published research — not a black-box API.
        </p>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100 flex items-start gap-4 hover:shadow-md transition-all duration-150 hover:-translate-y-0.5"
            >
              <div className="w-10 h-10 rounded-full bg-stone-100 flex items-center justify-center text-lg flex-shrink-0">
                {f.icon}
              </div>
              <div>
                <div className="font-bold text-stone-900 mb-1 text-sm">{f.title}</div>
                <div className="text-xs text-stone-500 leading-relaxed">{f.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-24">
        <p className="text-sm font-semibold text-red-600 mb-3">• Use cases</p>
        <h2 className="text-4xl font-black text-stone-900 tracking-tight mb-10">
          Works for every type of content
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {CATEGORIES.map((c) => (
            <div
              key={c.n}
              className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100 flex items-start gap-4 hover:shadow-md transition-all duration-150"
            >
              <span className="text-xs font-mono text-red-500 font-bold pt-0.5">/{c.n}</span>
              <div>
                <div className="font-bold text-stone-900 mb-1">{c.title}</div>
                <div className="text-sm text-stone-500">{c.desc}</div>
              </div>
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
