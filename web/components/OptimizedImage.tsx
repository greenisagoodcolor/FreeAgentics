import Image, { ImageProps } from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface OptimizedImageProps extends ImageProps {
  fallbackSrc?: string;
  aspectRatio?: number;
  showLoader?: boolean;
}

/**
 * Production-optimized image component with:
 * - Automatic blur placeholder generation
 * - Error handling with fallback
 * - Loading states
 * - Responsive sizing
 * - Lazy loading by default
 */
export function OptimizedImage({
  src,
  alt,
  className,
  fallbackSrc = "/images/placeholder.png",
  aspectRatio,
  showLoader = true,
  priority = false,
  quality = 85,
  sizes,
  ...props
}: OptimizedImageProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  // Generate responsive sizes if not provided
  const defaultSizes = sizes || "(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw";

  // Create blur data URL for placeholder
  const blurDataURL = createBlurPlaceholder();

  return (
    <div
      className={cn(
        "relative overflow-hidden",
        aspectRatio ? "aspect-ratio-container" : undefined,
        className,
      )}
      style={aspectRatio ? { paddingBottom: `${(1 / aspectRatio) * 100}%` } : undefined}
    >
      <Image
        src={error ? fallbackSrc : src}
        alt={alt}
        className={cn(
          "transition-opacity duration-300",
          isLoading && showLoader ? "opacity-0" : "opacity-100",
          aspectRatio ? "absolute inset-0 w-full h-full object-cover" : undefined,
        )}
        quality={quality}
        sizes={defaultSizes}
        priority={priority}
        placeholder={priority ? "blur" : "empty"}
        blurDataURL={priority ? blurDataURL : undefined}
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setError(true);
          setIsLoading(false);
        }}
        {...props}
      />

      {/* Loading skeleton */}
      {isLoading && showLoader && <div className="absolute inset-0 bg-gray-200 animate-pulse" />}
    </div>
  );
}

/**
 * Create a simple blur placeholder data URL
 * In production, this would be generated at build time
 */
function createBlurPlaceholder(): string {
  // Simple 1x1 pixel gray image as base64
  return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWEREiMxUf/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q==";
}

/**
 * Picture component for art-directed images
 */
export function OptimizedPicture({
  sources,
  alt,
  className,
  fallbackSrc = "/images/placeholder.png",
}: {
  sources: Array<{
    srcSet: string;
    media?: string;
    type?: string;
  }>;
  alt: string;
  className?: string;
  fallbackSrc?: string;
}) {
  return (
    <picture className={className}>
      {sources.map((source, index) => (
        <source key={index} srcSet={source.srcSet} media={source.media} type={source.type} />
      ))}
      <OptimizedImage
        src={sources[sources.length - 1]?.srcSet || fallbackSrc}
        alt={alt}
        className="w-full h-auto"
      />
    </picture>
  );
}

/**
 * Background image component with lazy loading
 */
export function BackgroundImage({
  src,
  className,
  children,
  overlay = false,
  parallax = false,
}: {
  src: string;
  className?: string;
  children?: React.ReactNode;
  overlay?: boolean;
  parallax?: boolean;
}) {
  const [loaded, setLoaded] = useState(false);

  // Preload image
  if (typeof window !== "undefined") {
    const img = new window.Image();
    img.src = src;
    img.onload = () => setLoaded(true);
  }

  return (
    <div
      className={cn(
        "relative bg-cover bg-center",
        parallax && "bg-fixed",
        !loaded && "bg-gray-200",
        className,
      )}
      style={loaded ? { backgroundImage: `url(${src})` } : undefined}
    >
      {overlay && <div className="absolute inset-0 bg-black bg-opacity-50" />}
      {children && <div className="relative z-10">{children}</div>}
    </div>
  );
}

/**
 * Avatar component with optimized loading
 */
export function Avatar({
  src,
  alt,
  size = "md",
  fallbackInitials,
  className,
}: {
  src?: string;
  alt: string;
  size?: "sm" | "md" | "lg" | "xl";
  fallbackInitials?: string;
  className?: string;
}) {
  const [error] = useState(false);

  const sizeClasses = {
    sm: "w-8 h-8 text-xs",
    md: "w-12 h-12 text-sm",
    lg: "w-16 h-16 text-base",
    xl: "w-24 h-24 text-lg",
  };

  if (!src || error) {
    return (
      <div
        className={cn(
          "rounded-full bg-gray-300 flex items-center justify-center font-medium text-gray-600",
          sizeClasses[size],
          className,
        )}
      >
        {fallbackInitials || alt.charAt(0).toUpperCase()}
      </div>
    );
  }

  return (
    <OptimizedImage
      src={src}
      alt={alt}
      width={size === "sm" ? 32 : size === "md" ? 48 : size === "lg" ? 64 : 96}
      height={size === "sm" ? 32 : size === "md" ? 48 : size === "lg" ? 64 : 96}
      className={cn("rounded-full object-cover", sizeClasses[size], className)}
      showLoader={false}
    />
  );
}
