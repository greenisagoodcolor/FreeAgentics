/** @type {import('next').NextConfig} */
const nextConfig = {
  // Performance optimizations
  compiler: {
    removeConsole: process.env.NODE_ENV === "production",
  },

  // Bundle optimization
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Optimize bundle splitting
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: "all",
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: "vendors",
            chunks: "all",
          },
          d3: {
            test: /[\\/]node_modules[\\/]d3/,
            name: "d3",
            chunks: "all",
          },
          recharts: {
            test: /[\\/]node_modules[\\/]recharts/,
            name: "recharts",
            chunks: "all",
          },
          framerMotion: {
            test: /[\\/]node_modules[\\/]framer-motion/,
            name: "framer-motion",
            chunks: "all",
          },
        },
      },
    };

    // Resolve path aliases
    config.resolve.alias = {
      ...config.resolve.alias,
      "@": require("path").resolve(__dirname, "."),
      "@/components": require("path").resolve(__dirname, "components"),
      "@/store": require("path").resolve(__dirname, "store"),
      "@/hooks": require("path").resolve(__dirname, "hooks"),
      "@/lib": require("path").resolve(__dirname, "lib"),
      "@/styles": require("path").resolve(__dirname, "styles"),
    };

    // Optimize for development
    if (dev) {
      config.watchOptions = {
        poll: 1000,
        aggregateTimeout: 300,
      };
    }

    // Production optimizations
    if (!dev) {
      // Tree shaking optimization
      config.optimization.usedExports = true;
      config.optimization.sideEffects = false;

      // Minimize bundle size
      config.resolve.alias = {
        ...config.resolve.alias,
        "react/jsx-runtime.js": "react/jsx-runtime",
        "react/jsx-dev-runtime.js": "react/jsx-dev-runtime",
      };
    }

    return config;
  },

  // Image optimization
  images: {
    domains: ["localhost"],
    formats: ["image/webp", "image/avif"],
  },

  // Headers for better caching
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
        ],
      },
    ];
  },

  // Enable source maps in development
  productionBrowserSourceMaps: false,

  // Reduce build output
  output: "standalone",

  // Enable React strict mode
  reactStrictMode: true,

  // Disable x-powered-by header
  poweredByHeader: false,

  // Compress static files
  compress: true,

  // Enable experimental features for better performance
  experimental: {
    optimizePackageImports: ["lucide-react", "framer-motion", "d3", "recharts"],
    // optimizeCss: true, // Disabled - requires critters package
    scrollRestoration: true,
    largePageDataBytes: 128 * 100, // 12.8KB
  },
};

module.exports = nextConfig;
