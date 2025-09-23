import React, { useEffect } from 'react';
import CodeSnippetViewer from './CodeSnippetViewer';

/**
 * Component that displays a code snippet with its associated visualization.
 * This creates a clear cause-effect relationship between code and its output.
 */
function CodeWithVisualization({ snippet }) {
  useEffect(() => {
    // Log component initialization with detailed snippet info
    console.debug('[CodeWithVisualization] Rendering with snippet:', {
      hasCode: !!snippet?.code,
      codeLength: snippet?.code?.length,
      language: snippet?.language,
      hasVisualization: !!snippet?.visualization,
      visualizationType: snippet?.visualization?.src ? 'src' : snippet?.visualization?.data ? 'data' : 'none',
      hasDescription: !!snippet?.visualization?.description
    });
  }, [snippet]);

  // Input validation with detailed error logging
  if (!snippet || typeof snippet !== 'object') {
    console.error('[CodeWithVisualization] Invalid snippet provided:', snippet);
    return null;
  }

  const { code, language, visualization } = snippet;
  
  // Check that we have code - the essential part
  if (!code) {
    console.warn('[CodeWithVisualization] Snippet missing code:', snippet);
    return null;
  }

  // Detailed visualization validation
  const hasValidVisualization = visualization && 
    (visualization.src || (visualization.data && visualization.format));

  if (visualization && !hasValidVisualization) {
    console.warn('[CodeWithVisualization] Visualization data is invalid:', visualization);
  }

  return (
    <div className="code-vis-container mb-6">
      {/* Code snippet first */}
      <CodeSnippetViewer snippet={{ code, language }} />
      
      {/* Visualization below the code if available and valid */}
      {hasValidVisualization && (
        <div className="visualization-container mt-3">
          <img
            src={visualization.src || `data:image/${visualization.format || 'png'};base64,${visualization.data}`}
            alt={visualization.description || "Visualization result"}
            title={visualization.description}
            className="rounded shadow-sm transition-opacity duration-300 max-w-full h-auto"
            loading="lazy"
            onError={(e) => {
              console.error(`[CodeWithVisualization] Error loading visualization:`, e);
              console.error(`[CodeWithVisualization] Failed visualization data:`, {
                srcLength: visualization.src?.length,
                dataLength: visualization.data?.length,
                format: visualization.format,
                description: visualization.description
              });
              e.target.style.display = 'none';
            }}
            onLoad={() => console.debug('[CodeWithVisualization] Visualization loaded successfully')}
          />
          {visualization.description && (
            <div className="text-sm text-gray-600 mt-1">{visualization.description}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default CodeWithVisualization;
