import { useState } from 'react';
import { DashboardLayout } from './layouts/DashboardLayout';
import { TrendScoutPage } from './pages/TrendScoutPage';
import { Construction } from 'lucide-react';

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-4">
      <div className="p-4 rounded-full bg-indigo-500/10 text-indigo-400 animate-pulse">
        <Construction className="w-12 h-12" />
      </div>
      <h2 className="text-3xl font-bold">{title}</h2>
      <p className="text-slate-400">This agent is currently under development.</p>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('trend-scout');

  return (
    <DashboardLayout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'trend-scout' && <TrendScoutPage />}
      {activeTab === 'video-creator' && <PlaceholderPage title="Video Creator Agent" />}
      {activeTab === 'orchestrator' && <PlaceholderPage title="Orchestrator Agent" />}
      {activeTab === 'settings' && <PlaceholderPage title="System Settings" />}
    </DashboardLayout>
  );
}

export default App;
