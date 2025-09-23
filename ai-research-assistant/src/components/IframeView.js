import React, { useState, useEffect, useRef, useMemo } from 'react';

function IframeView({ url }) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null); // 'x-frame-options', 'general', etc.
  const iframeRef = useRef(null);
  
  // List of domains known to block iframe embedding - wrapped in useMemo to avoid recreating on each render
  const knownRestrictedDomains = useMemo(() => [
    'researchgate.net',
    'linkedin.com',
    'scholar.google.com',
    'ieee.org',
    'academia.edu',
    'facebook.com',
    'twitter.com',
    'instagram.com'
  ], []);
  
  // Handle iframe load event
  const handleIframeLoad = () => {
    setIsLoading(false);
  };
  
  // Handle iframe error
  const handleIframeError = () => {
    const domain = new URL(url).hostname;
    const isKnownRestrictedDomain = knownRestrictedDomains.some(restrictedDomain => 
      domain.includes(restrictedDomain)
    );
    
    if (isKnownRestrictedDomain) {
      setErrorType('x-frame-options');
      setError(`${domain} prevents embedding in iframes for security reasons.`);
    } else {
      setErrorType('general');
      setError('Failed to load the requested page');
    }
    setIsLoading(false);
  };
  
  // Reset loading state when URL changes
  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setErrorType(null);
    
    // Preemptively check if the domain is known to block iframes
    try {
      const domain = new URL(url).hostname;
      const isKnownRestrictedDomain = knownRestrictedDomains.some(restrictedDomain => 
        domain.includes(restrictedDomain)
      );
      
      if (isKnownRestrictedDomain) {
        setErrorType('x-frame-options');
        setError(`${domain} likely prevents embedding in iframes for security reasons.`);
        setIsLoading(false);
      }
    } catch (e) {
      console.error("Error parsing URL:", e);
    }
  }, [url, knownRestrictedDomains]);
  
  // Open URL in a new tab/window
  const openInNewTab = () => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="h-full flex flex-col">
      {/* Browser-like URL bar */}
      <div className="px-4 py-2 border-b border-gray-200 flex items-center bg-gray-50">
        <div className="flex items-center flex-1 px-3 py-1.5 bg-white border border-gray-300 rounded shadow-sm">
          <svg className="w-4 h-4 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm text-gray-700 font-mono truncate">{url}</span>
        </div>
      </div>
      
      {/* Iframe container */}
      <div className="flex-1 relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
            <div className="flex flex-col items-center">
              <svg className="w-8 h-8 text-blue-500 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="mt-2 text-sm text-gray-600">Loading page...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-white z-10">
            <div className="max-w-md p-6 bg-red-50 rounded-lg shadow-sm">
              <div className="flex items-center mb-3">
                <svg className="w-6 h-6 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="text-lg font-medium text-red-800">
                  {errorType === 'x-frame-options' ? 'Website Embedding Restricted' : 'Error Loading Page'}
                </h3>
              </div>
              <p className="text-sm text-red-700">{error}</p>
              
              {errorType === 'x-frame-options' ? (
                <>
                  <p className="mt-3 text-sm text-gray-700">
                    Many websites set <code className="px-1 py-0.5 bg-gray-100 rounded">X-Frame-Options</code> to prevent
                    embedding for security reasons. This can't be bypassed from the browser.
                  </p>
                  <div className="mt-4 flex flex-col space-y-2">
                    <button
                      onClick={openInNewTab}
                      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
                    >
                      Open in New Tab
                    </button>
                    <div className="text-center text-xs text-gray-500">or</div>
                    <div className="border border-gray-300 rounded p-2 flex items-center bg-gray-50">
                      <input 
                        type="text" 
                        value={url} 
                        readOnly 
                        className="flex-1 bg-transparent border-none text-sm focus:outline-none" 
                        onClick={(e) => e.target.select()}
                      />
                      <button 
                        onClick={() => navigator.clipboard.writeText(url).then(() => alert('URL copied to clipboard'))} 
                        className="ml-2 p-1 text-gray-500 hover:text-gray-700"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <p className="mt-3 text-sm text-red-700">
                  This could be due to content security restrictions, network connectivity issues, or the page no longer exists.
                </p>
              )}
            </div>
          </div>
        )}
        
        <iframe 
          ref={iframeRef}
          src={url}
          className="w-full h-full border-none"
          onLoad={handleIframeLoad}
          onError={handleIframeError}
          title="Web Content"
          sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
        />
      </div>
      
      {/* Status bar */}
      <div className="border-t border-gray-200 px-4 py-1.5 text-xs text-gray-500 flex justify-between items-center bg-gray-50">
        <div>
          <div className="flex items-center">
            {isLoading ? (
              <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-1"></span>
            ) : (
              <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-1"></span>
            )}
            {isLoading ? 'Loading...' : 'Loaded'}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            iframe view
          </span>
        </div>
      </div>
    </div>
  );
}

export default IframeView;
