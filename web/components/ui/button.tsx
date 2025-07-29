import * as React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "sm" | "lg" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    const variants = {
      default: "bg-blue-600 text-white border-2 border-blue-700 hover:bg-blue-700 shadow-md font-semibold",
      destructive: "bg-red-600 text-white border-2 border-red-700 hover:bg-red-700 shadow-md font-semibold",
      outline: "border-2 border-gray-400 bg-white text-gray-900 hover:bg-gray-100 shadow-md font-semibold",
      secondary: "bg-gray-200 text-gray-900 border-2 border-gray-300 hover:bg-gray-300 shadow-md font-semibold",
      ghost: "text-gray-900 hover:bg-gray-200 font-semibold",
      link: "text-blue-600 underline-offset-4 hover:underline font-semibold",
    };

    const sizes = {
      default: "h-12 px-6 py-3 text-base",
      sm: "h-10 px-4 py-2 text-sm",
      lg: "h-14 px-8 py-4 text-lg",
      icon: "h-12 w-12 text-base",
    };

    return (
      <button
        className={`inline-flex items-center justify-center rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${variants[variant]} ${sizes[size]} ${className || ""}`}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button };
