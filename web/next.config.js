/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    optimizePackageImports: ['@reduxjs/toolkit', 'react-redux'],
  },
  webpack: (config, { isServer }) => {
    // Fix for Redux Toolkit module resolution issues
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }

    // Optimize Redux Toolkit imports
    config.resolve.alias = {
      ...config.resolve.alias,
      '@reduxjs/toolkit': require.resolve('@reduxjs/toolkit'),
      'react-redux': require.resolve('react-redux'),
    };

    return config;
  },
  transpilePackages: ['@reduxjs/toolkit', 'react-redux'],
};

module.exports = nextConfig;
