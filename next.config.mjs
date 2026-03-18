/** @type {import('next').NextConfig} */
const config = {
  output: 'standalone',
  experimental: {
    serverComponentsExternalPackages: ['better-sqlite3', '@xenova/transformers', 'onnxruntime-node', 'sharp', 'tesseract.js', 'pdf-parse', 'mammoth', 'puppeteer-core'],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push(
        'better-sqlite3', '@xenova/transformers', 'onnxruntime-node',
        'pdf-parse', 'mammoth', 'tesseract.js', 'sharp', 'encoding',
        'puppeteer-core',
      );
    }
    return config;
  },
};

export default config;
