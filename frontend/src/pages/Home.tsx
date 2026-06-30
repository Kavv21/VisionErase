import { Link } from 'react-router-dom'

const STEPS = [
  { n: '001', title: 'Upload footage', desc: 'Drop in any MP4, MOV, or WebM file up to 2 GB.' },
  { n: '002', title: 'Mark the object', desc: 'Paint over the object on the first frame — takes seconds.' },
  { n: '003', title: 'AI removes it', desc: 'SAM 2 segments it; XMem++ tracks it across every frame.' },
  { n: '004', title: 'Review result', desc: 'Preview the cleaned video before downloading.' },
  { n: '005', title: 'Download', desc: 'Get your file in the original format, quality preserved.' },
]

const FEATURES = [
  {
    title: 'SAM 2 Segmentation',
    desc: 'Meta\'s Segment Anything Model precisely isolates any object from a single painted frame.',
  },
  {
    title: 'Occlusion-aware tracking',
    desc: 'XMem++ re-identifies objects after they disappear behind other subjects.',
  },
  {
    title: 'Seamless long-video processing',
    desc: 'Video is split into parallel chunks and stitched with our BoundaryFusion model for temporal consistency.',
  },
  {
    title: 'Quality-checked output',
    desc: 'Automatic SSIM and PSNR scoring ensures every frame meets quality thresholds before delivery.',
  },
]

const STATS = [
  { label: 'Frames processed', value: '50M+' },
  { label: 'Avg. processing time', value: '< 3 min' },
  { label: 'Output accuracy', value: '97.4%' },
]

export default function Home() {
  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-16">
        <p className="text-sm font-semibold text-red-600 mb-6">• AI Video Editing</p>
        <h1 className="text-6xl md:text-7xl font-black text-stone-900 leading-[0.95] tracking-tight max-w-3xl mb-6">
          Remove anything from any video — automatically.
        </h1>
        <p className="text-lg text-stone-500 max-w-xl mb-10">
          Paint over an object once. VisionErase removes it from every frame, across the entire video, with no manual work.
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <Link
            to="/upload"
            className="px-6 py-3 bg-red-600 text-white font-semibold rounded-xl hover:bg-red-700 transition-all duration-150 hover:scale-[1.02]"
          >
            Try it free
          </Link>
          <Link to="/features" className="text-stone-600 font-medium hover:text-stone-900 transition-colors">
            See how it works →
          </Link>
        </div>
      </section>

      {/* Stats strip */}
      <section className="max-w-6xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {STATS.map((s) => (
            <div
              key={s.label}
              className="bg-white rounded-2xl p-6 shadow-sm border border-stone-100 hover:shadow-md transition-shadow duration-200"
            >
              <div className="text-3xl font-black text-stone-900 mb-1">{s.value}</div>
              <div className="text-sm text-stone-500">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <p className="text-sm font-semibold text-red-600 mb-3">• How it works</p>
        <h2 className="text-4xl md:text-5xl font-black text-stone-900 tracking-tight mb-12">
          Five steps to a clean video
        </h2>
        <div className="space-y-4">
          {STEPS.map((step) => (
            <div
              key={step.n}
              className="flex items-start gap-6 bg-white rounded-2xl p-6 shadow-sm border border-stone-100 hover:shadow-md transition-all duration-150 hover:-translate-y-0.5"
            >
              <div className="flex-shrink-0 flex items-center gap-3">
                <span className="text-xs font-mono text-red-500 font-bold">/{step.n}</span>
                <span className="text-stone-400">›</span>
              </div>
              <div>
                <div className="font-bold text-stone-900 mb-1">{step.title}</div>
                <div className="text-sm text-stone-500">{step.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <p className="text-sm font-semibold text-red-600 mb-3">• Features</p>
        <h2 className="text-4xl md:text-5xl font-black text-stone-900 tracking-tight mb-12">
          Built on real CV research
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="bg-white rounded-2xl p-7 shadow-sm border border-stone-100 hover:shadow-md transition-all duration-150 hover:-translate-y-0.5"
            >
              <h3 className="font-bold text-stone-900 mb-2">{f.title}</h3>
              <p className="text-sm text-stone-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA band */}
      <section className="bg-stone-900 py-20 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight mb-6">
            Ready to erase?
          </h2>
          <p className="text-stone-400 mb-8 text-lg">No watermarks. No editing skills required.</p>
          <Link
            to="/upload"
            className="inline-block px-8 py-4 bg-red-600 text-white font-bold rounded-xl hover:bg-red-500 transition-all duration-150 hover:scale-[1.02]"
          >
            Get started free
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-6xl mx-auto px-6 py-10 flex items-center justify-between text-sm text-stone-400 border-t border-stone-200">
        <span className="font-bold text-stone-900">VisionErase</span>
        <span>© 2026 VisionErase. All rights reserved.</span>
      </footer>
    </div>
  )
}
