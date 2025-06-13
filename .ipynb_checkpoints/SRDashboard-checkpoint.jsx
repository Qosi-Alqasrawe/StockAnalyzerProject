import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Target, Activity, BarChart3, Zap } from 'lucide-react';



// لو الكومبوننت في ملف آخر، استورده هنا

/**
 * SRDashboard Component
 * Renders an interactive Support & Resistance dashboard with three tabs:
 * - Overview: Pivot support/resistance metrics
 * - SR Zones: Highlight active zones
 * - Fibonacci: Retracement levels and visual
 */
export default function SRDashboard({ data, currentPrice }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [animatedValues, setAnimatedValues] = useState({});

  // Animate pivot metrics on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValues({
        long_support: data.pivot_levels.long_support,
        long_resistance: data.pivot_levels.long_resistance,
        short_support: data.pivot_levels.short_support,
        short_resistance: data.pivot_levels.short_resistance
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [data.pivot_levels]);

  const getPriceColor = (price) => {
    if (price > currentPrice) return 'text-red-400';
    if (price < currentPrice) return 'text-green-400';
    return 'text-yellow-400';
  };

  const getPriceBg = (price) => {
    if (price > currentPrice) return 'bg-red-500/10 border-red-500/30';
    if (price < currentPrice) return 'bg-green-500/10 border-green-500/30';
    return 'bg-yellow-500/10 border-yellow-500/30';
  };

  // Card components
  const PriceCard = ({ title, value, icon: Icon, type }) => {
    const distance = Math.abs(value - currentPrice);
    const percentage = ((distance / currentPrice) * 100).toFixed(2);
    return (
      <div
        className={`p-6 rounded-xl border-2 transition-all duration-300 hover:scale-105 hover:shadow-lg ${getPriceBg(value)}`}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Icon className={`w-5 h-5 ${getPriceColor(value)}`} />
            <span className="text-gray-300 font-medium">{title}</span>
          </div>
          <div
            className={`px-2 py-1 rounded-lg text-xs font-bold ${
              type === 'resistance'
                ? 'bg-red-500/20 text-red-300'
                : 'bg-green-500/20 text-green-300'
            }`}
          >
            {type}
          </div>
        </div>
        <div className={`text-2xl font-bold ${getPriceColor(value)} mb-2`}>
          {value.toFixed(2)}
        </div>
        <div className="text-sm text-gray-400">
          {distance.toFixed(2)} ({percentage}%){' '}
          {value > currentPrice ? 'above' : 'below'}
        </div>
      </div>
    );
  };

  const ZoneCard = ({ zone, index }) => {
    const [min, max] = zone;
    const isInZone = currentPrice >= min && currentPrice <= max;
    return (
      <div
        className={`p-4 rounded-lg border transition-all duration-300 hover:scale-102 ${
          isInZone
            ? 'bg-yellow-500/20 border-yellow-500 shadow-yellow-500/20 shadow-lg'
            : 'bg-gray-800/50 border-gray-700'
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-gray-200">Zone {index + 1}</span>
          {isInZone && (
            <span className="px-2 py-1 bg-yellow-500 text-black text-xs rounded-full font-bold animate-pulse">
              ACTIVE
            </span>
          )}
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-green-400">Support: {min.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-red-400">Resistance: {max.toFixed(2)}</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                isInZone
                  ? 'bg-gradient-to-r from-green-400 to-red-400'
                  : 'bg-gray-600'
              }`}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      </div>
    );
  };

  const FibCard = ({ name, value }) => {
    const level = parseFloat(name.split('_')[1]);
    const color =
      level < 50
        ? 'text-green-400'
        : level === 50
        ? 'text-yellow-400'
        : 'text-red-400';
    const bgColor =
      level < 50
        ? 'bg-green-500/10'
        : level === 50
        ? 'bg-yellow-500/10'
        : 'bg-red-500/10';
    return (
      <div className={`p-4 rounded-lg ${bgColor} border border-gray-700 hover:border-gray-600 transition-all duration-300`}>
        <div className="flex justify-between items-center">
          <span className="text-gray-300 text-sm">{name}</span>
          <span className={`font-bold ${color}`}>{value.toFixed(2)}</span>
        </div>
        <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
          <div
            className={`h-1 rounded-full bg-gradient-to-r from-transparent ${color.replace('text-', 'to-')}`}
            style={{ width: `${level}%` }}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-4">
          <Activity className="w-8 h-8 text-blue-400" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Support & Resistance Analysis
          </h1>
        </div>

        {/* Current Price Display */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-gray-300">Current Price</span>
            </div>
            <div className="text-3xl font-bold text-green-400">{currentPrice.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-2 mb-6 bg-gray-800/30 rounded-lg p-1">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'zones', label: 'SR Zones', icon: Target },
          { id: 'fibonacci', label: 'Fibonacci', icon: Zap }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Dynamic Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <PriceCard
            title="Long Support"
            value={animatedValues.long_support || 0}
            icon={TrendingUp}
            type="support"
          />
          <PriceCard
            title="Long Resistance"
            value={animatedValues.long_resistance || 0}
            icon={TrendingDown}
            type="resistance"
          />
          <PriceCard
            title="Short Support"
            value={animatedValues.short_support || 0}
            icon={TrendingUp}
            type="support"
          />
          <PriceCard
            title="Short Resistance"
            value={animatedValues.short_resistance || 0}
            icon={TrendingDown}
            type="resistance"
          />
        </div>
      )}

      {activeTab === 'zones' && (
        <div className="space-y-4">
          <div className="flex items-center space-x-2 mb-4">
            <Target className="w-5 h-5 text-blue-400" />
            <h2 className="text-xl font-semibold">Support & Resistance Zones</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {data.sr_zones.map((zone, idx) => (
              <ZoneCard key={idx} zone={zone} index={idx} />
            ))}
          </div>
        </div>
      )}

      {activeTab === 'fibonacci' && (
        <div className="space-y-4">
          <div className="flex items-center space-x-2 mb-4">
            <Zap className="w-5 h-5 text-yellow-400" />
            <h2 className="text-xl font-semibold">Fibonacci Levels</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(data.fib_levels).map(([name, value]) => (
              <FibCard key={name} name={name} value={value} />
            ))}
          </div>

          <div className="mt-8 p-6 bg-gray-800/30 rounded-xl border border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-center">Fibonacci Retracement Visual</h3>
            <div className="space-y-2">
              {Object.entries(data.fib_levels).map(([name, value]) => {
                const level = parseFloat(name.split('_')[1]);
                return (
                  <div key={name} className="flex items-center space-x-3">
                    <div className="w-12 text-sm text-gray-400">{level}%</div>
                    <div className="flex-1 bg-gray-700 rounded-full h-6 relative overflow-hidden">
