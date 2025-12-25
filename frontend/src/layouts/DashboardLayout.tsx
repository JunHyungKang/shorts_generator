import { useState } from 'react';
import { TrendingUp, Video, BrainCircuit, Settings, Menu, X, Rocket } from 'lucide-react';

interface DashboardLayoutProps {
    children: React.ReactNode;
    activeTab: string;
    onTabChange: (tab: string) => void;
}

export function DashboardLayout({ children, activeTab, onTabChange }: DashboardLayoutProps) {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const navItems = [
        { id: 'trend-scout', label: 'Trend Scout', icon: TrendingUp },
        { id: 'video-creator', label: 'Video Creator', icon: Video },
        { id: 'orchestrator', label: 'Orchestrator', icon: BrainCircuit },
        { id: 'settings', label: 'Settings', icon: Settings },
    ];

    return (
        <div className="min-h-screen bg-[#0a0a0c] text-slate-200">
            {/* Sidebar (Desktop) */}
            <aside className="fixed inset-y-0 left-0 z-50 w-64 hidden lg:flex flex-col border-r border-white/5 bg-[#0a0a0c]/80 backdrop-blur-xl">
                <div className="h-16 flex items-center px-6 border-b border-white/5">
                    <div className="flex items-center gap-2 text-indigo-400">
                        <Rocket className="w-6 h-6" />
                        <span className="font-bold text-lg bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                            ShortsGen
                        </span>
                    </div>
                </div>

                <nav className="flex-1 p-4 space-y-1">
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => onTabChange(item.id)}
                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === item.id
                                    ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                                }`}
                        >
                            <item.icon className="w-5 h-5" />
                            {item.label}
                        </button>
                    ))}
                </nav>

                <div className="p-4 border-t border-white/5">
                    <div className="px-3 py-2 rounded-lg bg-slate-900/50 border border-white/5">
                        <p className="text-xs text-slate-500 font-mono">Agent Status</p>
                        <div className="flex items-center gap-2 mt-1.5">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                            </span>
                            <span className="text-xs font-semibold text-emerald-500">Online</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Mobile Header */}
            <header className="lg:hidden h-16 flex items-center justify-between px-4 border-b border-white/5 bg-[#0a0a0c]/80 backdrop-blur-xl sticky top-0 z-40">
                <div className="flex items-center gap-2 text-indigo-400">
                    <Rocket className="w-5 h-5" />
                    <span className="font-bold text-lg">ShortsGen</span>
                </div>
                <button
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    className="p-2 -mr-2 text-slate-400 hover:text-white"
                >
                    {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                </button>
            </header>

            {/* Mobile Menu */}
            {isMobileMenuOpen && (
                <div className="fixed inset-0 z-30 lg:hidden bg-[#0a0a0c] pt-16">
                    <nav className="p-4 space-y-1">
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => {
                                    onTabChange(item.id);
                                    setIsMobileMenuOpen(false);
                                }}
                                className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === item.id
                                        ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                                    }`}
                            >
                                <item.icon className="w-5 h-5" />
                                {item.label}
                            </button>
                        ))}
                    </nav>
                </div>
            )}

            {/* Main Content */}
            <main className="lg:pl-64 min-h-screen">
                <div className="max-w-7xl mx-auto px-4 py-8 lg:px-8">
                    {children}
                </div>
            </main>
        </div>
    );
}
