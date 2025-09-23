import React from 'react';

/**
 * LoadingIndicator component displays an animated progress indicator
 * @param {Object} props - Component props
 * @param {string} props.type - Type of indicator: 'bar', 'spinner', 'dots', 'pulse' (default: 'spinner')
 * @param {string} props.size - Size of the indicator: 'small', 'medium', 'large' (default: 'medium')
 * @param {string} props.color - Primary color of the indicator (default: '#1a5fb4')
 * @param {string} props.text - Optional text to display with the indicator
 * @param {boolean} props.fullScreen - Whether to display the indicator in fullscreen overlay
 * @param {number} props.progress - Progress value (0-100) for bar type indicators
 */
function LoadingIndicator({ 
  type = 'spinner', 
  size = 'medium', 
  color = '#1a5fb4', 
  text = '',
  fullScreen = false,
  progress = -1 
}) {
  // Size mappings in pixels
  const sizeMap = {
    small: { container: 16, spinner: 16, bar: 4, dots: 8, pulse: 8 },
    medium: { container: 24, spinner: 24, bar: 6, dots: 10, pulse: 12 },
    large: { container: 40, spinner: 40, bar: 8, dots: 14, pulse: 18 }
  };
  
  const selectedSize = sizeMap[size] || sizeMap.medium;
  
  // Base styles
  const styles = {
    container: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    },
    fullScreenOverlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(255, 255, 255, 0.8)',
      zIndex: 9999,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    text: {
      marginTop: 10,
      fontSize: size === 'small' ? 12 : size === 'large' ? 16 : 14,
      color: '#333',
    },
    // Spinner styles
    spinner: {
      width: selectedSize.spinner,
      height: selectedSize.spinner,
      border: `${selectedSize.spinner / 8}px solid rgba(0, 0, 0, 0.1)`,
      borderRadius: '50%',
      borderTop: `${selectedSize.spinner / 8}px solid ${color}`,
      animation: 'spin 1s linear infinite',
    },
    // Bar styles
    barContainer: {
      width: selectedSize.container * 5,
      height: selectedSize.bar,
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
      borderRadius: selectedSize.bar / 2,
      overflow: 'hidden',
    },
    bar: {
      height: '100%',
      backgroundColor: color,
      borderRadius: selectedSize.bar / 2,
      transition: 'width 0.3s ease',
      width: progress >= 0 && progress <= 100 ? `${progress}%` : '0%',
      animation: progress < 0 ? 'barIndeterminate 2s ease-in-out infinite' : 'none',
    },
    // Dots styles
    dotsContainer: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: selectedSize.dots / 2,
    },
    dot: {
      width: selectedSize.dots,
      height: selectedSize.dots,
      borderRadius: '50%',
      backgroundColor: color,
    },
    // Pulse styles
    pulse: {
      width: selectedSize.pulse,
      height: selectedSize.pulse,
      borderRadius: '50%',
      backgroundColor: color,
      animation: 'pulse 1.5s ease-in-out infinite',
    }
  };

  // Keyframes are added as a <style> tag to the component
  const keyframes = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    @keyframes barIndeterminate {
      0% { width: 0%; left: 0; }
      50% { width: 70%; left: 10%; }
      100% { width: 0%; left: 100%; }
    }
    @keyframes dotPulse1 {
      0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
      40% { transform: scale(1); opacity: 1; }
    }
    @keyframes dotPulse2 {
      0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
      40% { transform: scale(1); opacity: 1; }
    }
    @keyframes dotPulse3 {
      0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
      40% { transform: scale(1); opacity: 1; }
    }
    @keyframes pulse {
      0% { transform: scale(0.8); opacity: 0.5; }
      50% { transform: scale(1); opacity: 1; }
      100% { transform: scale(0.8); opacity: 0.5; }
    }
  `;

  // Render different types of indicators
  const renderLoadingIndicator = () => {
    switch (type) {
      case 'bar':
        return (
          <div style={styles.barContainer}>
            <div style={styles.bar} />
          </div>
        );
      case 'dots':
        return (
          <div style={styles.dotsContainer}>
            <div style={{...styles.dot, animation: 'dotPulse1 1.4s infinite ease-in-out'}} />
            <div style={{...styles.dot, animation: 'dotPulse2 1.4s infinite ease-in-out 0.2s'}} />
            <div style={{...styles.dot, animation: 'dotPulse3 1.4s infinite ease-in-out 0.4s'}} />
          </div>
        );
      case 'pulse':
        return <div style={styles.pulse} />;
      case 'spinner':
      default:
        return <div style={styles.spinner} />;
    }
  };

  const containerStyle = fullScreen
    ? styles.fullScreenOverlay
    : styles.container;

  return (
    <>
      <style>{keyframes}</style>
      <div style={containerStyle}>
        <div style={styles.container}>
          {renderLoadingIndicator()}
          {text && <div style={styles.text}>{text}</div>}
        </div>
      </div>
    </>
  );
}

export default LoadingIndicator;
