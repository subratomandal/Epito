'use client';

import { useEffect, useState, useCallback } from 'react';

interface Props {
  onComplete: () => void;
}

export default function BrandedSplash({ onComplete }: Props) {
  const [phase, setPhase] = useState(0);
  const [done, setDone] = useState(false);

  const finish = useCallback(() => {
    if (done) return;
    setDone(true);
    onComplete();
  }, [done, onComplete]);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      finish();
      return;
    }

    // 1.3s total: dots(0-350) → converge+E(350-700) → text(700-1050) → glow+fade(1050-1300)
    const t1 = setTimeout(() => setPhase(1), 350);
    const t2 = setTimeout(() => setPhase(2), 700);
    const t3 = setTimeout(() => setPhase(3), 1050);
    const t4 = setTimeout(finish, 1300);

    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); };
  }, [finish]);

  if (done) return null;

  return (
    <div
      onClick={finish}
      style={{
        position: 'fixed', inset: 0, zIndex: 99999,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'hsl(var(--background))', cursor: 'pointer', overflow: 'hidden',
      }}
    >
      <style>{`
        .sp { position: relative; width: 280px; height: 60px; display: flex; align-items: center; justify-content: center; }

        .dt {
          position: absolute; width: 5px; height: 5px; border-radius: 50%;
          background: hsl(var(--primary)); opacity: 0;
        }
        .p0 .dt { animation: de 0.3s cubic-bezier(0.16,1,0.3,1) forwards; }
        .p0 .dt:nth-child(1) { --x: -35px; --y: -18px; animation-delay: 0s; }
        .p0 .dt:nth-child(2) { --x: 30px; --y: -22px; animation-delay: 0.05s; }
        .p0 .dt:nth-child(3) { --x: -25px; --y: 20px; animation-delay: 0.1s; }
        .p0 .dt:nth-child(4) { --x: 38px; --y: 14px; animation-delay: 0.15s; }
        .p0 .dt:nth-child(5) { --x: -12px; --y: -30px; animation-delay: 0.2s; }

        @keyframes de {
          0% { opacity:0; transform: scale(0); }
          40% { opacity:1; transform: translate(var(--x),var(--y)) scale(1.3); }
          100% { opacity:0.7; transform: translate(calc(var(--x)*0.6),calc(var(--y)*0.6)) scale(0.7); }
        }

        .p1 .dt { animation: dc 0.3s cubic-bezier(0.34,1.56,0.64,1) forwards; opacity:0.7; }
        @keyframes dc { to { opacity:0; transform: translate(0,0) scale(0); } }

        .le {
          position: absolute; left:50%; top:50%; transform:translate(-50%,-50%);
          font: 700 40px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
          letter-spacing: -0.02em; color: hsl(var(--foreground)); opacity:0; user-select:none;
        }
        .p1 .le { animation: ea 0.3s cubic-bezier(0.16,1,0.3,1) 0.1s forwards; }
        @keyframes ea {
          from { opacity:0; transform:translate(-50%,-50%) scale(0.6); }
          to { opacity:1; transform:translate(-50%,-50%) scale(1); }
        }

        .lr {
          position: absolute; left:calc(50% + 11px); top:50%; transform:translateY(-50%);
          font: 700 40px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
          letter-spacing: -0.02em; color: hsl(var(--foreground)); user-select:none;
          display:flex;
        }
        .lr span { opacity:0; display:inline-block; }
        .p2 .lr span { animation: cw 0.2s cubic-bezier(0.16,1,0.3,1) forwards; }
        .p2 .lr span:nth-child(1) { animation-delay: 0s; }
        .p2 .lr span:nth-child(2) { animation-delay: 0.04s; }
        .p2 .lr span:nth-child(3) { animation-delay: 0.08s; }
        .p2 .lr span:nth-child(4) { animation-delay: 0.12s; }
        @keyframes cw { from{opacity:0;transform:translateX(-3px)} to{opacity:1;transform:translateX(0)} }

        .p3 { animation: sf 0.25s ease-out forwards; }
        .p3 .le, .p3 .lr span { opacity:1; }
        @keyframes sf { 0%{opacity:1} 50%{opacity:1;filter:brightness(1.1)} 100%{opacity:0} }
      `}</style>

      <div className={`sp p${phase}`}>
        <div className="dt"/><div className="dt"/><div className="dt"/><div className="dt"/><div className="dt"/>
        <div className="le">E</div>
        <div className="lr"><span>p</span><span>i</span><span>t</span><span>o</span></div>
      </div>
    </div>
  );
}
