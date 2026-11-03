/** Tailwind config for the offline build (see README). Keep in sync with templates/base.html. */
module.exports = {
  content: ['./templates/**/*.html'],
  safelist: ['bg-low', 'bg-high', 'text-low', 'text-mid', 'text-high'],
  theme: {
    extend: {
      colors: {
        brand: { 50: '#eef6f3', 100: '#d5e9e2', 600: '#23705f', 700: '#1d5c50', 800: '#174a41', 900: '#123a33' },
        ink: '#18221f', paper: '#f3f5f2',
        low: '#2f7a46', mid: '#a8690f', high: '#b23a2e'
      },
      fontFamily: {
        display: ['Archivo', 'Helvetica Neue', 'Arial', 'sans-serif'],
        sans: ['Public Sans', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'Menlo', 'Consolas', 'monospace']
      }
    }
  }
}
