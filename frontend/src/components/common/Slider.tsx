/**
 * Reusable Slider Component
 */

import React from 'react';

interface SliderProps {
  label?: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  showValue?: boolean;
  formatValue?: (value: number) => string;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled = false,
  showValue = true,
  formatValue = (v) => v.toString(),
  className = '',
}) => {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`space-y-2 ${className}`}>
      {(label || showValue) && (
        <div className="flex justify-between items-center">
          {label && <span className="text-sm text-secondary">{label}</span>}
          {showValue && (
            <span className="text-sm font-mono text-primary">
              {formatValue(value)}
            </span>
          )}
        </div>
      )}
      
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="
            w-full h-2 bg-surface rounded-lg appearance-none cursor-pointer
            disabled:opacity-50 disabled:cursor-not-allowed
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:bg-primary
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:bg-primary
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:border-0
            [&::-moz-range-thumb]:cursor-pointer
          "
          style={{
            background: `linear-gradient(to right, #64ffda ${percentage}%, #1a1a2e ${percentage}%)`,
          }}
        />
      </div>
      
      <div className="flex justify-between text-xs text-secondary/50">
        <span>{formatValue(min)}</span>
        <span>{formatValue(max)}</span>
      </div>
    </div>
  );
};

export default Slider;

