import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';

function CodeSnippetViewer({ snippet, initialCollapsed = true }) {
  const [codeState, setCodeState] = useState({
    collapsed: initialCollapsed, // Use the prop to determine initial state
    wrap: false,
    darkTheme: true,
    copied: false,
  });

  // Validation check
  if (!snippet || typeof snippet !== 'object' || !snippet.code) {
    console.warn("CodeSnippetViewer received invalid snippet data:", snippet);
    return null;
  }

  // Extract necessary props
  const { language, code } = snippet;

  // Determine what code to display based on collapsed state
  const getDisplayCode = () => {
    if (!codeState.collapsed) return code;

    const lines = code.split('\n');
    if (lines.length <= 15) return code;

    return lines.slice(0, 15).join('\n') + '\n// ...';
  };

  // Check if this is a special case of minimal content
  const isMinimalContent = code.trim().length < 15 || code.trim().split(/\s+/).length <= 2;
  const singleWord = code.trim().split(/\s+/).length === 1;

  // Determine the language to use for syntax highlighting
  let syntaxLanguage = language;
  if (!language && isMinimalContent) {
    syntaxLanguage = 'plaintext';
  }


  // Code block toolbar toggle handlers
  const toggleCollapse = () => {
    setCodeState(prev => ({ ...prev, collapsed: !prev.collapsed }));
  };

  const toggleWrap = () => {
    setCodeState(prev => ({ ...prev, wrap: !prev.wrap }));
  };

  const toggleTheme = () => {
    setCodeState(prev => ({ ...prev, darkTheme: !prev.darkTheme }));
  };

  const handleCodeCopy = () => {
    // Always copy the full code regardless of display state
    navigator.clipboard.writeText(code)
      .then(() => {
        setCodeState(prev => ({ ...prev, copied: true }));
        setTimeout(() => {
          setCodeState(prev => ({ ...prev, copied: false }));
        }, 2000);
      })
      .catch(err => {
        console.error("Failed to copy code: ", err);
      });
  };

  // Define CSS styles for the component
  const styles = {
    container: {
      maxWidth: '670px',
      width: '100%',
      overflow: 'hidden',
      borderRadius: '4px',
      border: '1px solid #e2e8f0',
      backgroundColor: '#f8fafc'
    },
    toolbar: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '4px 10px',
      borderBottom: '1px solid #e2e8f0',
      backgroundColor: '#f1f5f9'
    },
    toolbarLeft: {
      display: 'flex',
      alignItems: 'center'
    },
    toolbarRight: {
      display: 'flex',
      gap: '8px'
    },
    language: {
      fontSize: '12px',
      color: '#64748b',
      fontFamily: 'monospace'
    },
    button: {
      padding: '2px 8px',
      fontSize: '12px',
      border: '1px solid #cbd5e1',
      borderRadius: '3px',
      backgroundColor: 'white',
      color: '#64748b',
      cursor: 'pointer'
    },
    activeButton: {
      backgroundColor: '#e2e8f0'
    },
    successButton: {
      backgroundColor: '#bbf7d0',
      borderColor: '#86efac',
      color: '#166534'
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.toolbar}>
        <div style={styles.toolbarLeft}>
          {language && <span style={styles.language}>{language}</span>}
        </div>
        <div style={styles.toolbarRight}>
          <button
            style={{
              ...styles.button,
              ...(codeState.collapsed ? styles.activeButton : {})
            }}
            onClick={toggleCollapse}
          >
            {codeState.collapsed ? 'Expand' : 'Collapse'}
          </button>
          <button
            style={{
              ...styles.button,
              ...(codeState.wrap ? styles.activeButton : {})
            }}
            onClick={toggleWrap}
          >
            {codeState.wrap ? 'Unwrap' : 'Wrap'}
          </button>
          <button
            style={{
              ...styles.button,
              ...(!codeState.darkTheme ? styles.activeButton : {})
            }}
            onClick={toggleTheme}
          >
            {codeState.darkTheme ? 'Light' : 'Dark'}
          </button>
          <button
            style={{
              ...styles.button,
              ...(codeState.copied ? styles.successButton : {})
            }}
            onClick={handleCodeCopy}
          >
            {codeState.copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>
      <div style={{ maxHeight: codeState.collapsed ? '400px' : 'none', overflow: codeState.collapsed ? 'hidden' : 'auto' }}>
        <SyntaxHighlighter
          language={syntaxLanguage}
          style={codeState.darkTheme ? tomorrow : vs}
          customStyle={{
            margin: 0,
            padding: isMinimalContent ? '0.75rem' : '1rem', // Smaller padding for minimal content
            fontSize: '0.875rem',
            lineHeight: '1.6',
            backgroundColor: codeState.darkTheme ? '#2d2d2d' : '#f9fafb',
            whiteSpace: codeState.wrap ? 'pre-wrap' : 'pre',
            wordBreak: codeState.wrap ? 'break-word' : 'normal',
            overflow: 'auto',
            maxWidth: '100%',
            minHeight: isMinimalContent ? '2.5rem' : 'auto', // Ensure small blocks have min height
            borderRadius: '0 0 4px 4px',
            fontFamily: 'Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
          }}
          showLineNumbers={false}
          wrapLines={true}
          codeTagProps={{
            style: {
              fontFamily: 'Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
            }
          }}
        >
          {getDisplayCode()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

export default CodeSnippetViewer; 