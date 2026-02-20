import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

// Flat Card Component (Replacing GlassCard)
export function Card({ children, className, noPadding = false, ...props }) {
    return (
        <div
            className={cn(
                "bg-white border border-slate-200 rounded-xl shadow-sm text-slate-800",
                !noPadding && "p-6",
                className
            )}
            {...props}
        >
            {children}
        </div>
    );
}

// Clean Button Component
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
                "btn-base",
                variant === 'primary' ? 'btn-primary' : 'btn-secondary',
                className
            )}
            {...props}
        >
            {Icon && <Icon className="w-4 h-4 mr-2" />}
            {children}
        </button>
    );
}

// Clean Input Component
export function Input({ label, className, ...props }) {
    return (
        <div className="flex flex-col gap-1.5 w-full">
            {label && <label className="text-sm font-medium text-slate-700">{label}</label>}
            <input
                className={cn("input-base", className)}
                {...props}
            />
        </div>
    );
}

// Clean Badge Component
export function Badge({ children, variant = 'gray', className }) {
    const variants = {
        gray: 'bg-slate-100 text-slate-700 border border-slate-200',
        blue: 'bg-blue-50 text-blue-700 border border-blue-200',
        green: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
        purple: 'bg-purple-50 text-purple-700 border border-purple-200',
    };

    return (
        <span className={cn(
            "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium tracking-wide",
            variants[variant] || variants.gray,
            className
        )}>
            {children}
        </span>
    );
}
