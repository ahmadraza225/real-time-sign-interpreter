import { Link, NavLink, Outlet } from "react-router-dom";

type LayoutProps = {
  highContrast: boolean;
  onToggleContrast: () => void;
};

const navItems = [
  ["/", "Home"],
  ["/interpreter", "Live Interpreter"],
  ["/results", "Results"],
  ["/analytics", "Analytics"],
  ["/about", "About"],
] as const;

export default function Layout({ highContrast, onToggleContrast }: LayoutProps) {
  return (
    <div className={highContrast ? "high-contrast min-h-screen" : "min-h-screen"}>
      <header className="border-b border-slate-800 bg-slate-900/90">
        <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4" aria-label="Primary">
          <Link className="text-lg font-bold" to="/">
            Real-Time Sign Interpreter
          </Link>
          <ul className="flex flex-wrap items-center gap-3 text-sm">
            {navItems.map(([path, label]) => (
              <li key={path}>
                <NavLink className={({ isActive }) => (isActive ? "font-semibold text-cyan-300" : "text-slate-300")} to={path}>
                  {label}
                </NavLink>
              </li>
            ))}
            <li>
              <button
                type="button"
                onClick={onToggleContrast}
                className="rounded border border-slate-600 px-2 py-1"
                aria-pressed={highContrast}
                aria-label="Toggle high contrast mode"
              >
                {highContrast ? "Standard" : "High Contrast"}
              </button>
            </li>
          </ul>
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">
        <Outlet />
      </main>
      <footer className="border-t border-slate-800 px-4 py-6 text-center text-sm text-slate-400">
        Breaking the communication barrier, one gesture at a time.
      </footer>
    </div>
  );
}
