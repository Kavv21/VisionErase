const USE_CASES = [
  { title: 'Remove a passerby', color: 'from-blue-200 to-blue-400', desc: 'Street photography without photobombers.' },
  { title: 'Erase a logo', color: 'from-purple-200 to-purple-400', desc: 'Clean product footage for neutral distribution.' },
  { title: 'Clean background clutter', color: 'from-green-200 to-green-400', desc: 'Remove distracting objects from any scene.' },
  { title: 'Remove text overlays', color: 'from-amber-200 to-amber-400', desc: 'Strip watermarks, burned-in subtitles, or labels.' },
  { title: 'Erase camera equipment', color: 'from-pink-200 to-pink-400', desc: 'Remove a mic boom or reflector from frame.' },
  { title: 'Clean archival footage', color: 'from-teal-200 to-teal-400', desc: 'Restore legacy video by removing artifacts.' },
]

export default function Work() {
  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <section className="max-w-6xl mx-auto px-6 pt-24 pb-16">
        <p className="text-sm font-semibold text-red-600 mb-6">• Work</p>
        <h1 className="text-6xl md:text-7xl font-black text-stone-900 leading-[0.95] tracking-tight max-w-3xl mb-6">
          See it in action
        </h1>
        <p className="text-lg text-stone-500 max-w-xl">
          From busy street shots to clean product videos — VisionErase handles every use case automatically.
        </p>
      </section>

      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {USE_CASES.map((uc) => (
            <div
              key={uc.title}
              className="group relative overflow-hidden rounded-2xl aspect-video bg-white shadow-sm border border-stone-100 hover:shadow-lg transition-all duration-200 hover:-translate-y-1 cursor-pointer"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${uc.color} opacity-60 group-hover:opacity-80 transition-opacity duration-200`} />
              <div className="absolute inset-0 flex flex-col justify-end p-5">
                <div className="bg-white/90 backdrop-blur rounded-xl p-4">
                  <div className="font-bold text-stone-900 text-sm mb-1">{uc.title}</div>
                  <div className="text-xs text-stone-500">{uc.desc}</div>
                </div>
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
