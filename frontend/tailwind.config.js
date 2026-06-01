/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#030303',
        surface: '#0d0d0d',
        primary: '#00f3ff',     // Tactical Cyan
        secondary: '#ffb000',   // Alert Amber
        danger: '#ff003c',      // Critical Red
        success: '#00ff88',     // Safe Green
        glass: 'rgba(13, 13, 13, 0.75)',
        glass_border: 'rgba(0, 243, 255, 0.15)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Share Tech Mono', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glowPulse: {
          '0%': { boxShadow: '0 0 10px rgba(0, 243, 255, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 243, 255, 0.4)' },
        }
      }
    },
  },
  plugins: [],
}
