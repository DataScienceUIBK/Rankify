import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

// Glass Card Component
export function GlassCard({ children, className, hover = true, ...props }) {
    return (
        <div
            className={cn(
                "glass-panel w-full",
                !hover && "hover:bg-panel hover:shadow-none hover:transform-none",
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

// Gradient Button Component
export function Button({
    children,
    variant = 'primary',
    className,
    icon: Icon,
    ...props
}) {
    return (
        <button
            className={cn(
                "btn",
                variant === 'primary' ? 'btn-primary' : 'btn-secondary',
                className
            )}
            {...props}
        >
            {Icon && <Icon className="w-4 h-4" />}
            {children}
        </button>
    );
}

// Glowing Input Component
export function Input({ label, className, ...props }) {
    return (
        <div className="input-group w-full">
            {label && <label className="input-label">{label}</label>}
            <input
                className={cn("glass-input", className)}
                {...props}
            />
        </div>
    );
}

// Badge Component
export function Badge({ children, color = 'blue', className }) {
    return (
        <span className={cn(`badge badge-${color}`, className)}>
            {children}
        </span>
    );
}
