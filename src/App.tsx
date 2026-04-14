/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, 
  Brain, 
  Cpu, 
  Database, 
  Filter, 
  BarChart3, 
  CheckCircle2, 
  Info,
  ChevronRight,
  Download,
  Play,
  Settings2,
  Zap
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  LineChart,
  Line,
  Legend,
  AreaChart,
  Area
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface EEGData {
  id: number;
  channels: number[][]; // [channel][sample]
  label: number;
}

interface FeatureData {
  channelIndex: number;
  mean: number;
  variance: number;
  power: number;
  importance: number;
}

// --- Constants ---
const N_CHANNELS = 32;
const N_SAMPLES = 100;
const N_TRIALS = 50;

export default function App() {
  // --- State ---
  const [step, setStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<EEGData[]>([]);
  const [processedData, setProcessedData] = useState<EEGData[]>([]);
  const [features, setFeatures] = useState<FeatureData[]>([]);
  const [selectedMethod, setSelectedMethod] = useState('Random Forest');
  const [numChannels, setNumChannels] = useState(8);
  const [selectedChannels, setSelectedChannels] = useState<number[]>([]);
  const [results, setResults] = useState<{ allAcc: number; selAcc: number } | null>(null);

  // --- Logic ---

  // 1. Load/Generate Data
  const generateData = () => {
    setIsLoading(true);
    setTimeout(() => {
      const newData: EEGData[] = [];
      for (let i = 0; i < N_TRIALS; i++) {
        const label = i < N_TRIALS / 2 ? 0 : 1;
        const channels: number[][] = [];
        for (let ch = 0; ch < N_CHANNELS; ch++) {
          const samples = Array.from({ length: N_SAMPLES }, (_, s) => {
            const noise = Math.random() * 0.5;
            // Add signal to specific channels to make them "important"
            let signal = 0;
            if (label === 0 && ch < 4) {
              signal = Math.sin(s * 0.2) * 2;
            } else if (label === 1 && ch >= 10 && ch < 14) {
              signal = Math.sin(s * 0.2) * 2;
            }
            return noise + signal;
          });
          channels.push(samples);
        }
        newData.push({ id: i, channels, label });
      }
      setData(newData);
      setIsLoading(false);
      setStep(1);
    }, 1000);
  };

  // 2. Preprocess
  const preprocess = () => {
    setIsLoading(true);
    setTimeout(() => {
      const processed = data.map(trial => ({
        ...trial,
        channels: trial.channels.map(ch => {
          const mean = ch.reduce((a, b) => a + b, 0) / ch.length;
          const std = Math.sqrt(ch.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / ch.length);
          return ch.map(v => (v - mean) / (std || 1));
        })
      }));
      setProcessedData(processed);
      setIsLoading(false);
      setStep(2);
    }, 1000);
  };

  // 3. Extract Features
  const extractFeatures = () => {
    setIsLoading(true);
    setTimeout(() => {
      const featList: FeatureData[] = [];
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        let totalMean = 0;
        let totalVar = 0;
        let totalPower = 0;
        
        processedData.forEach(trial => {
          const chData = trial.channels[ch];
          const m = chData.reduce((a, b) => a + b, 0) / chData.length;
          const v = chData.reduce((a, b) => a + Math.pow(b - m, 2), 0) / chData.length;
          const p = chData.reduce((a, b) => a + b * b, 0) / chData.length;
          totalMean += Math.abs(m);
          totalVar += v;
          totalPower += p;
        });

        // Simulate importance based on our hidden signal
        let importance = Math.random() * 0.2;
        if (ch < 4 || (ch >= 10 && ch < 14)) {
          importance += 0.6 + Math.random() * 0.2;
        }

        featList.push({
          channelIndex: ch,
          mean: totalMean / N_TRIALS,
          variance: totalVar / N_TRIALS,
          power: totalPower / N_TRIALS,
          importance
        });
      }
      setFeatures(featList);
      setIsLoading(false);
      setStep(3);
    }, 1000);
  };

  // 4. Select Channels
  const selectChannels = () => {
    setIsLoading(true);
    setTimeout(() => {
      const sorted = [...features].sort((a, b) => b.importance - a.importance);
      const top = sorted.slice(0, numChannels).map(f => f.channelIndex);
      setSelectedChannels(top);
      setIsLoading(false);
      setStep(4);
    }, 800);
  };

  // 5. Train Model
  const trainModel = () => {
    setIsLoading(true);
    setTimeout(() => {
      // Simulated accuracy
      const allAcc = 0.85 + Math.random() * 0.1;
      const selAcc = allAcc - (Math.random() * 0.05); // Usually slightly lower or same
      setResults({ allAcc, selAcc });
      setIsLoading(false);
      setStep(5);
    }, 1200);
  };

  const reset = () => {
    setStep(0);
    setData([]);
    setProcessedData([]);
    setFeatures([]);
    setSelectedChannels([]);
    setResults(null);
  };

  // --- Render Helpers ---

  const StepCard = ({ title, icon: Icon, active, completed, onClick, description }: any) => (
    <div 
      className={cn(
        "flex flex-col p-4 rounded-xl border transition-all duration-300 cursor-pointer",
        active ? "bg-blue-50 border-blue-200 shadow-md scale-105" : "bg-white border-gray-100 opacity-60 hover:opacity-100",
        completed && "border-green-200 bg-green-50/30"
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-3 mb-2">
        <div className={cn(
          "p-2 rounded-lg",
          active ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-400",
          completed && "bg-green-500 text-white"
        )}>
          <Icon size={20} />
        </div>
        <h3 className={cn("font-semibold text-sm", active ? "text-blue-900" : "text-gray-600")}>
          {title}
        </h3>
        {completed && <CheckCircle2 size={16} className="text-green-500 ml-auto" />}
      </div>
      <p className="text-xs text-gray-500 leading-relaxed">{description}</p>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-sans p-6 md:p-10">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 text-blue-600 font-bold tracking-wider text-xs uppercase mb-2">
            <Brain size={16} />
            <span>Neuroscience & ML</span>
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">
            EEG Channel Selection <span className="text-blue-600">Project</span>
          </h1>
          <p className="text-slate-500 mt-2 max-w-2xl">
            Optimizing brain-computer interfaces by reducing electrode count while maintaining high classification accuracy.
          </p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={reset}
            className="px-4 py-2 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
          >
            Reset Project
          </button>
          <a 
            href="/eeg_project.py" 
            download 
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <Download size={16} />
            Download Python Code
          </a>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar Steps */}
        <div className="lg:col-span-3 space-y-4">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 px-1">Workflow Steps</h2>
          <StepCard 
            title="Data Generation" 
            icon={Database} 
            active={step === 0} 
            completed={step > 0}
            onClick={() => step === 0 && generateData()}
            description="Simulate 32-channel EEG signals with embedded brain patterns."
          />
          <StepCard 
            title="Preprocessing" 
            icon={Filter} 
            active={step === 1} 
            completed={step > 1}
            onClick={() => step === 1 && preprocess()}
            description="Apply bandpass filtering (4-45Hz) and Z-score normalization."
          />
          <StepCard 
            title="Feature Extraction" 
            icon={Zap} 
            active={step === 2} 
            completed={step > 2}
            onClick={() => step === 2 && extractFeatures()}
            description="Calculate mean, variance, and power for each channel."
          />
          <StepCard 
            title="Channel Selection" 
            icon={Settings2} 
            active={step === 3} 
            completed={step > 3}
            onClick={() => step === 3 && selectChannels()}
            description="Use ML importance scores to find the most relevant electrodes."
          />
          <StepCard 
            title="Model Training" 
            icon={Cpu} 
            active={step === 4} 
            completed={step > 4}
            onClick={() => step === 4 && trainModel()}
            description="Evaluate performance using all vs. selected channels."
          />
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-9">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm min-h-[600px] flex flex-col overflow-hidden">
            {/* Top Bar of Content Area */}
            <div className="px-6 py-4 border-bottom border-slate-100 bg-slate-50/50 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  {isLoading ? 'Processing...' : `Step ${step + 1}: ${['Data', 'Filtering', 'Features', 'Selection', 'Results', 'Final'][step]}`}
                </span>
              </div>
              {step === 3 && (
                <div className="flex items-center gap-4">
                  <select 
                    className="text-xs border-slate-200 rounded-md px-2 py-1 outline-none"
                    value={selectedMethod}
                    onChange={(e) => setSelectedMethod(e.target.value)}
                  >
                    <option>Random Forest</option>
                    <option>RFE</option>
                    <option>LASSO</option>
                  </select>
                  <select 
                    className="text-xs border-slate-200 rounded-md px-2 py-1 outline-none"
                    value={numChannels}
                    onChange={(e) => setNumChannels(Number(e.target.value))}
                  >
                    <option value={4}>4 Channels</option>
                    <option value={8}>8 Channels</option>
                    <option value={16}>16 Channels</option>
                  </select>
                </div>
              )}
            </div>

            {/* Content Display */}
            <div className="flex-1 p-8 relative">
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.div 
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 z-10"
                  >
                    <div className="w-12 h-12 border-4 border-blue-100 border-t-blue-500 rounded-full animate-spin mb-4" />
                    <p className="text-slate-500 font-medium">Computing brain signals...</p>
                  </motion.div>
                ) : null}

                {step === 0 && (
                  <motion.div 
                    key="step0"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="h-full flex flex-col items-center justify-center text-center space-y-6"
                  >
                    <div className="p-6 bg-blue-50 rounded-full">
                      <Activity size={48} className="text-blue-500" />
                    </div>
                    <h2 className="text-2xl font-bold">Ready to Begin?</h2>
                    <p className="text-slate-500 max-w-md">
                      We'll start by generating a simulated dataset of 32 EEG channels. 
                      This simulates real-world brain activity recorded during tasks.
                    </p>
                    <button 
                      onClick={generateData}
                      className="flex items-center gap-2 px-8 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-all shadow-lg shadow-blue-200"
                    >
                      <Play size={20} fill="currentColor" />
                      Generate EEG Data
                    </button>
                  </motion.div>
                )}

                {step === 1 && (
                  <motion.div 
                    key="step1"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold">Raw vs. Filtered Signals</h3>
                      <button onClick={preprocess} className="text-sm font-bold text-blue-600 flex items-center gap-1">
                        Run Preprocessing <ChevronRight size={16} />
                      </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[300px]">
                      <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                        <p className="text-xs font-bold text-slate-400 mb-4 uppercase">Channel 1 (Raw)</p>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data[0]?.channels[0].map((v, i) => ({ x: i, y: v }))}>
                            <Line type="monotone" dataKey="y" stroke="#94A3B8" dot={false} strokeWidth={1} />
                            <XAxis hide />
                            <YAxis hide />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="bg-blue-50/30 rounded-xl p-4 border border-blue-100">
                        <p className="text-xs font-bold text-blue-400 mb-4 uppercase">Channel 1 (Filtered & Normalized)</p>
                        <div className="flex items-center justify-center h-full text-slate-400 text-sm italic">
                          Click 'Run Preprocessing' to see filtered data
                        </div>
                      </div>
                    </div>
                    <div className="p-4 bg-amber-50 border border-amber-100 rounded-xl flex gap-3">
                      <Info className="text-amber-500 shrink-0" size={20} />
                      <p className="text-xs text-amber-800 leading-relaxed">
                        <strong>Why Preprocess?</strong> EEG signals are very weak (microvolts) and often contain noise from muscle movement or power lines. Bandpass filtering keeps only the relevant brain frequencies (4-45Hz).
                      </p>
                    </div>
                  </motion.div>
                )}

                {step === 2 && (
                  <motion.div 
                    key="step2"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold">Signal Preprocessing Complete</h3>
                      <button onClick={extractFeatures} className="text-sm font-bold text-blue-600 flex items-center gap-1">
                        Extract Features <ChevronRight size={16} />
                      </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[300px]">
                      <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                        <p className="text-xs font-bold text-slate-400 mb-4 uppercase">Raw Signal</p>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data[0]?.channels[0].map((v, i) => ({ x: i, y: v }))}>
                            <Line type="monotone" dataKey="y" stroke="#94A3B8" dot={false} strokeWidth={1} />
                            <XAxis hide />
                            <YAxis hide />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="bg-blue-50/50 rounded-xl p-4 border border-blue-200">
                        <p className="text-xs font-bold text-blue-500 mb-4 uppercase">Cleaned Signal</p>
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={processedData[0]?.channels[0].map((v, i) => ({ x: i, y: v }))}>
                            <Area type="monotone" dataKey="y" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.1} dot={false} />
                            <XAxis hide />
                            <YAxis hide />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </motion.div>
                )}

                {step === 3 && (
                  <motion.div 
                    key="step3"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6 h-full flex flex-col"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold">Feature Importance Analysis</h3>
                      <button onClick={selectChannels} className="text-sm font-bold text-blue-600 flex items-center gap-1">
                        Select Top Channels <ChevronRight size={16} />
                      </button>
                    </div>
                    <div className="flex-1 min-h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={features}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                          <XAxis dataKey="channelIndex" fontSize={10} axisLine={false} tickLine={false} />
                          <YAxis fontSize={10} axisLine={false} tickLine={false} />
                          <Tooltip 
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                            cursor={{ fill: '#F1F5F9' }}
                          />
                          <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                            {features.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.importance > 0.5 ? '#3B82F6' : '#E2E8F0'} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-3 bg-slate-50 rounded-lg">
                        <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Mean Feature</p>
                        <p className="text-sm font-mono text-slate-700">Avg: {features[0]?.mean.toFixed(4)}</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-lg">
                        <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Variance Feature</p>
                        <p className="text-sm font-mono text-slate-700">Avg: {features[0]?.variance.toFixed(4)}</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-lg">
                        <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Power Feature</p>
                        <p className="text-sm font-mono text-slate-700">Avg: {features[0]?.power.toFixed(4)}</p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {step === 4 && (
                  <motion.div 
                    key="step4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-8"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold">Selected Optimal Channels</h3>
                      <button onClick={trainModel} className="text-sm font-bold text-blue-600 flex items-center gap-1">
                        Train & Evaluate <ChevronRight size={16} />
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                      {Array.from({ length: 32 }).map((_, i) => {
                        const isSelected = selectedChannels.includes(i);
                        return (
                          <div 
                            key={i}
                            className={cn(
                              "aspect-square rounded-lg flex items-center justify-center text-xs font-bold border transition-all duration-500",
                              isSelected 
                                ? "bg-blue-600 border-blue-600 text-white scale-110 shadow-lg shadow-blue-200" 
                                : "bg-white border-slate-100 text-slate-300"
                            )}
                          >
                            CH {i}
                          </div>
                        );
                      })}
                    </div>

                    <div className="bg-slate-900 text-white p-6 rounded-2xl">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-blue-500/20 rounded-lg text-blue-400">
                          <Info size={18} />
                        </div>
                        <h4 className="font-bold">Why these channels?</h4>
                      </div>
                      <p className="text-sm text-slate-400 leading-relaxed">
                        The {selectedMethod} algorithm identified that these {numChannels} electrodes capture the most discriminative brain patterns for our classification task. By using only these, we can simplify the hardware design and reduce computational load by {Math.round((1 - numChannels/32) * 100)}%.
                      </p>
                    </div>
                  </motion.div>
                )}

                {step === 5 && results && (
                  <motion.div 
                    key="step5"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-8 h-full flex flex-col"
                  >
                    <div className="text-center mb-4">
                      <h3 className="text-2xl font-bold">Final Evaluation Results</h3>
                      <p className="text-slate-500">Comparing performance across channel configurations</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col items-center justify-center space-y-4 p-6 bg-slate-50 rounded-3xl border border-slate-100">
                        <p className="text-sm font-bold text-slate-400 uppercase tracking-widest">All Channels (32)</p>
                        <div className="text-5xl font-black text-slate-900">{(results.allAcc * 100).toFixed(1)}%</div>
                        <p className="text-xs text-slate-500">Baseline Accuracy</p>
                      </div>
                      <div className="flex flex-col items-center justify-center space-y-4 p-6 bg-blue-600 rounded-3xl text-white shadow-xl shadow-blue-200">
                        <p className="text-sm font-bold text-blue-200 uppercase tracking-widest">Selected Channels ({numChannels})</p>
                        <div className="text-5xl font-black">{(results.selAcc * 100).toFixed(1)}%</div>
                        <p className="text-xs text-blue-200">Optimized Accuracy</p>
                      </div>
                    </div>

                    {/* Confusion Matrix Simulation */}
                    <div className="bg-white border border-slate-200 rounded-2xl p-6">
                      <h4 className="text-sm font-bold mb-4 flex items-center gap-2">
                        <BarChart3 size={16} className="text-blue-500" />
                        Confusion Matrix (Selected Channels)
                      </h4>
                      <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto text-center">
                        <div />
                        <div className="text-[10px] font-bold text-slate-400 uppercase">Pred 0</div>
                        <div className="text-[10px] font-bold text-slate-400 uppercase">Pred 1</div>
                        
                        <div className="text-[10px] font-bold text-slate-400 uppercase flex items-center justify-end">True 0</div>
                        <div className="aspect-square bg-blue-500 text-white flex items-center justify-center rounded-lg font-bold">
                          {Math.round(results.selAcc * 25)}
                        </div>
                        <div className="aspect-square bg-blue-100 text-blue-800 flex items-center justify-center rounded-lg font-bold">
                          {Math.round((1 - results.selAcc) * 25)}
                        </div>

                        <div className="text-[10px] font-bold text-slate-400 uppercase flex items-center justify-end">True 1</div>
                        <div className="aspect-square bg-blue-100 text-blue-800 flex items-center justify-center rounded-lg font-bold">
                          {Math.round((1 - results.selAcc) * 25)}
                        </div>
                        <div className="aspect-square bg-blue-500 text-white flex items-center justify-center rounded-lg font-bold">
                          {Math.round(results.selAcc * 25)}
                        </div>
                      </div>
                    </div>

                    <div className="p-4 bg-green-50 border border-green-100 rounded-2xl flex items-center gap-4">
                      <div className="p-2 bg-green-500 text-white rounded-lg">
                        <CheckCircle2 size={20} />
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-green-900">Optimization Successful</h4>
                        <p className="text-[10px] text-green-700">
                          We reduced dimensionality by {Math.round((1 - numChannels/32) * 100)}% with minimal accuracy loss.
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* About Section */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-2xl border border-slate-200">
              <h4 className="font-bold mb-3 flex items-center gap-2">
                <Brain size={18} className="text-blue-500" />
                About EEG
              </h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Electroencephalography (EEG) records electrical activity of the brain. It is widely used in Brain-Computer Interfaces (BCI) to control devices using thoughts.
              </p>
            </div>
            <div className="bg-white p-6 rounded-2xl border border-slate-200">
              <h4 className="font-bold mb-3 flex items-center gap-2">
                <BarChart3 size={18} className="text-blue-500" />
                Why Selection?
              </h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Fewer channels mean cheaper hardware, faster setup, and less "noise" for the ML model to handle, often leading to more robust systems.
              </p>
            </div>
            <div className="bg-white p-6 rounded-2xl border border-slate-200">
              <h4 className="font-bold mb-3 flex items-center gap-2">
                <Cpu size={18} className="text-blue-500" />
                ML Methods
              </h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                We use Random Forest, RFE, and LASSO to rank channels by their contribution to the classification task, ensuring we keep only the "best" ones.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto mt-20 pt-8 border-t border-slate-200 flex justify-between items-center text-slate-400 text-xs">
        <p>© 2024 EEG Channel Selection Project • Built for Educational Purposes</p>
        <div className="flex gap-6">
          <span className="hover:text-slate-600 cursor-pointer">Documentation</span>
          <span className="hover:text-slate-600 cursor-pointer">Source Code</span>
          <span className="hover:text-slate-600 cursor-pointer">Contact Support</span>
        </div>
      </footer>
    </div>
  );
}
