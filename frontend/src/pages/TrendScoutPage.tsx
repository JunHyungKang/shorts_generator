import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Sparkles, TrendingUp, Search, Youtube, Loader2, AlertCircle } from 'lucide-react';

interface AnalysisResponse {
    report: string;
}

export function TrendScoutPage() {
    const [loading, setLoading] = useState(false);
    const [report, setReport] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        setReport(null);

        // TODO: Environment Variable or Config for API URL
        const API_URL = 'http://localhost:8000/analyze';

        try {
            const response = await axios.post<AnalysisResponse>(API_URL, {
                query: "Analyze the current YouTube trends for Korean seniors (60-70s) and suggest a video topic."
            });
            setReport(response.data.report);
        } catch (err) {
            console.error("Trend Analysis Error:", err); // Enhanced Logging
            if (axios.isAxiosError(err)) {
                if (!err.response) {
                    setError(`Network Error: Could not connect to ${API_URL}. Is the backend running?`);
                } else {
                    setError(`Server Error (${err.response.status}): ${err.response.data?.detail || err.message}`);
                }
            } else {
                setError("An unexpected error occurred.");
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-12">
            {/* Header Section */}
            <section className="text-center space-y-6 pt-8">
                <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight">
                    Trend <span className="text-indigo-400">Scout</span>
                </h2>
                <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
                    Autonomous AI agent that monitors web searches and validates YouTube views
                    to uncover the hottest topics for the 60-70s demographic.
                </p>

                <div className="pt-4">
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
                                Start Discovery
                            </>
                        )}
                    </button>
                </div>
            </section>

            {/* Progress Steps */}
            {loading && (
                <div className="max-w-2xl mx-auto grid grid-cols-3 gap-4">
                    {[
                        { icon: Search, label: "Web Discovery", active: true },
                        { icon: Youtube, label: "YouTube Verification", active: true },
                        { icon: Sparkles, label: "Report Gen", active: true },
                    ].map((step, idx) => (
                        <div key={idx} className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/5 border border-white/10 animate-pulse">
                            <step.icon className="w-5 h-5 text-indigo-400" />
                            <span className="text-xs font-medium text-slate-300 text-center">{step.label}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="max-w-2xl mx-auto p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3 text-red-200">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <p className="text-sm font-medium">{error}</p>
                </div>
            )}

            {/* Result Report */}
            {report && !loading && (
                <section className="max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-8 duration-700">
                    <div className="p-8 rounded-3xl bg-slate-900/50 backdrop-blur-xl border border-white/10 shadow-2xl ring-1 ring-white/5">
                        <div className="flex items-center gap-3 mb-8 pb-6 border-b border-white/10">
                            <div className="p-2.5 rounded-xl bg-indigo-500/20 text-indigo-400">
                                <TrendingUp className="w-6 h-6" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold">Trend Analysis Report</h3>
                                <p className="text-slate-400 text-sm">Target: Korean Seniors (60-70s)</p>
                            </div>
                        </div>

                        <div className="prose prose-invert prose-lg max-w-none prose-headings:text-indigo-200 prose-a:text-cyan-400 prose-strong:text-white prose-li:text-slate-300">
                            <ReactMarkdown>{report}</ReactMarkdown>
                        </div>
                    </div>
                </section>
            )}
        </div>
    );
}
