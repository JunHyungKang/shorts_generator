import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Sparkles, TrendingUp, Search, Youtube, Loader2, AlertCircle } from 'lucide-react';

interface AnalysisResponse {
  report: string;
}

function App() {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const response = await axios.post<AnalysisResponse>('http://localhost:8000/analyze', {
        query: "Analyze the current YouTube trends for Korean seniors (60-70s) and suggest a video topic."
      });
      setReport(response.data.report);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.message);
      } else {
        setError("Failed to connect to the analysis service.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-white selection:bg-indigo-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-white/5 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-indigo-400" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              ShortsInsight
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 border border-white/10">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              System Active
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <section className="text-center mb-16 space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-medium">
            <Sparkles className="w-4 h-4" />
            <span>Powered by Deep Agents & OpenRouter</span>
          </div>

          <h2 className="text-5xl md:text-6xl font-extrabold tracking-tight">
            Discover What's Trending <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-cyan-400">
              For Korean Seniors
            </span>
          </h2>

          <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Our autonomous AI agent monitors web searches and validates YouTube views
            to uncover the hottest topics for the 60-70s demographic.
          </p>

          <div className="pt-8">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="group relative inline-flex items-center gap-3 px-8 py-4 bg-white text-slate-900 rounded-full font-bold text-lg hover:bg-indigo-50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_-5px_rgba(255,255,255,0.5)]"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                  Analyzing Trends...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 text-indigo-600 group-hover:scale-110 transition-transform" />
                  Start Trend Analysis
                </>
              )}
            </button>
          </div>
        </section>

        {/* Status / Steps (Visible when loading) */}
        {loading && (
          <div className="max-w-2xl mx-auto mb-16 grid grid-cols-3 gap-4">
            {[
              { icon: Search, label: "Web Discovery", active: true },
              { icon: Youtube, label: "YouTube Verification", active: true },
              { icon: Sparkles, label: "Insight Generation", active: true },
            ].map((step, idx) => (
              <div key={idx} className="flex flex-col items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10 animate-pulse">
                <step.icon className="w-6 h-6 text-indigo-400" />
                <span className="text-sm font-medium text-slate-300">{step.label}</span>
              </div>
            ))}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-12 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3 text-red-200">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        )}

        {/* Report Display */}
        {report && !loading && (
          <section className="animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="relative p-8 rounded-3xl bg-slate-900/50 backdrop-blur-xl border border-white/10 shadow-2xl ring-1 ring-white/5">
              {/* Report Header */}
              <div className="flex items-center gap-3 mb-8 pb-6 border-b border-white/10">
                <div className="p-3 rounded-xl bg-indigo-500/20 text-indigo-400">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold">Trend Analysis Report</h3>
                  <p className="text-slate-400 text-sm">Generated by Trend Deep Agent</p>
                </div>
              </div>

              {/* Markdown Content */}
              <div className="prose prose-invert prose-lg max-w-none prose-headings:text-indigo-200 prose-a:text-cyan-400 prose-strong:text-white">
                <ReactMarkdown>{report}</ReactMarkdown>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
