/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        bg: '#FFFFFF',
        surface: '#F7F8F8',
        text: '#0F172A',
        accent: '#DA7756',
        border: '#E5E7EB',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '6px',
        md: '6px', // Explicitly set md if you use rounded-md often
      },
      spacing: {
        // Example: Define specific spacing values if needed, otherwise rely on defaults scaled appropriately.
        // You might want to align defaults closer to a 24px grid, e.g., 6: '1.5rem' (24px)
        '6': '1.5rem', // 24px
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)', // Keep or adjust default shadow-sm
      }
    },
  },
  plugins: [],
} 