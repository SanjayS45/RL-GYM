/**
 * Reusable Card Component
 */

import React from 'react';

interface CardProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  variant?: 'default' | 'bordered' | 'elevated';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  className?: string;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  subtitle,
  icon,
  action,
  variant = 'default',
  padding = 'md',
  className = '',
  onClick,
}) => {
  const baseStyles = 'rounded-xl transition-all duration-200';
  
  const variantStyles = {
    default: 'bg-surface',
    bordered: 'bg-surface border border-primary/20',
    elevated: 'bg-surface shadow-lg shadow-primary/5',
  };
  
  const paddingStyles = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };
  
  const clickableStyles = onClick
    ? 'cursor-pointer hover:border-primary/40 hover:shadow-lg hover:shadow-primary/10'
    : '';
  
  return (
    <div
      className={`
        ${baseStyles}
        ${variantStyles[variant]}
        ${paddingStyles[padding]}
        ${clickableStyles}
        ${className}
      `}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {(title || subtitle || icon || action) && (
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {icon && (
              <div className="text-primary text-xl">
                {icon}
              </div>
            )}
            <div>
              {title && (
                <h3 className="text-lg font-semibold text-primary">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="text-sm text-secondary">
                  {subtitle}
                </p>
              )}
            </div>
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
};

// Card Section for grouping content
interface CardSectionProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const CardSection: React.FC<CardSectionProps> = ({
  children,
  title,
  className = '',
}) => {
  return (
    <div className={`pt-4 border-t border-primary/10 first:pt-0 first:border-t-0 ${className}`}>
      {title && (
        <h4 className="text-sm font-medium text-secondary mb-3">
          {title}
        </h4>
      )}
      {children}
    </div>
  );
};

export default Card;

