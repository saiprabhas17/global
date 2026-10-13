import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    restoreMocks: true,
    unstubGlobals: true,
    coverage: {
      provider: 'v8',
      exclude: [
        'node_modules/**',
        'coverage/**',
        // Build tooling configs & app entrypoint — no logic to cover
        'postcss.config.js',
        'tailwind.config.js',
        'src/main.tsx',
        'src/vite-env.d.ts',
        // Decorative canvas art components — no logic-bearing branches,
        // animation math only. Excluded per documented policy (audit P1e);
        // they are exercised visually, not unit-testable in jsdom.
        'src/components/AsciiGlobe.tsx',
        'src/components/AsciiReveal.tsx',
        'src/components/AsciiEqualizer.tsx',
        'src/components/AsciiTicker.tsx',
        'src/components/AsciiSpinner.tsx',
        // 3D timeline tunnel — canvas/WebGL animation core, no API logic.
        'src/components/TimelineTunnel.tsx',
        // Force-simulation / whole-corpus canvas renderers: pure visual
        // layout & draw loops without API or decision logic (documented
        // exclusion policy — visually exercised, not unit-testable here).
        'src/components/GraphVisualizer.tsx',
        'src/components/FullCorpusGraph.tsx',
        // Arcade component driven by a requestAnimationFrame game loop.
        'src/components/MasalaLab.tsx',
      ],
      // Thresholds enforced from CI:
      // lines/functions=60, branches=50, statements=60
    },
  },
});
