/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // GitHub-inspired dark theme for research tools
        'canvas': '#010409',
        'surface': {
          DEFAULT: '#0d1117',
          subtle: '#161b22',
          muted: '#21262d',
        },
        'border': {
          DEFAULT: '#30363d',
          muted: '#21262d',
        },
        'fg': {
          DEFAULT: '#c9d1d9',
          muted: '#8b949e',
          subtle: '#484f58',
        },
        'accent': {
          blue: '#58a6ff',
          green: '#3fb950',
          purple: '#a371f7',
          orange: '#d29922',
          red: '#f85149',
          cyan: '#79c0ff',
        },
      },
      fontFamily: {
        'mono': ['SF Mono', 'Fira Code', 'Consolas', 'monospace'],
        'sans': ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
      },
      animation: {
        'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}
