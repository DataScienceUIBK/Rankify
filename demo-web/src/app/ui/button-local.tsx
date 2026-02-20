import * as React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "default" | "outline" | "ghost" | "secondary" | "destructive";
    size?: "default" | "sm" | "lg" | "icon";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className = "", variant = "default", size = "default", ...props }, ref) => {
        const base =
            "inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:pointer-events-none disabled:opacity-50";

        const variants: Record<string, string> = {
            default: "bg-indigo-600 text-white shadow hover:bg-indigo-700",
            outline: "border border-slate-200 bg-white shadow-sm hover:bg-slate-50 hover:text-slate-900",
            ghost: "hover:bg-slate-100 hover:text-slate-900",
            secondary: "bg-slate-100 text-slate-900 shadow-sm hover:bg-slate-200",
            destructive: "bg-red-500 text-white shadow-sm hover:bg-red-600",
        };

        const sizes: Record<string, string> = {
            default: "h-9 px-4 py-2",
            sm: "h-8 rounded-md px-3 text-xs",
            lg: "h-10 rounded-md px-8",
            icon: "h-9 w-9",
        };

        return (
            <button
                ref={ref}
                className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
                {...props}
            />
        );
    }
);
Button.displayName = "Button";
