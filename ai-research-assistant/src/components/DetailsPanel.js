import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import FinalReport from './FinalReport';
import { generateDetailsHtml } from '../utils/helpers';
import Tippy from '@tippyjs/react';
import CodeSnippetViewer from './CodeSnippetViewer';

// Helper function to extract report title (moved from ContentPanel.js)
const extractReportTitle = (content) => {
  if (!content) return null;
  
  // Check for HTML h1 title
  const h1Match = content.match(/<h1>(.*?)<\/h1>/i);
  if (h1Match && h1Match[1]) {
    const h1Title = h1Match[1].trim();
    // Skip if the title is just the search query
    if (!h1Title.toLowerCase().includes('what is') && 
        !h1Title.toLowerCase().includes('show me')) {
      return h1Title;
    }
  }
  
  // Try to find a title in the first few lines of the report
  const lines = content.split('\n').slice(0, 30);
  
  // First, look for specific title patterns that indicate a formal report title
  for (const line of lines) {
    const cleanLine = line.trim();
    // Look for titles with colons that describe frameworks, technologies, etc.
    if (cleanLine.match(/^\w+:\s+[A-Z]/i) || 
        cleanLine.match(/^[A-Z][\w\s]+:\s+[A-Z]/i)) {
      return cleanLine;
    }
  }
  
  // Then look for other title patterns
  for (const line of lines) {
    const cleanLine = line.trim();
    if (cleanLine.startsWith('# ')) {
      const title = cleanLine.replace(/^# /, '');
      // Skip if the title is just the search query
      if (!title.toLowerCase().includes('what is') && 
          !title.toLowerCase().includes('show me')) {
        return title;
      }
    }
    if (cleanLine.match(/^Profile of/i) || 
        cleanLine.match(/^Analysis of/i) ||
        cleanLine.match(/^Research on/i) ||
        cleanLine.match(/^State-of-the-Art/i) ||
        cleanLine.match(/^Overview of/i) ||
        cleanLine.match(/^Introduction to/i) ||
        cleanLine.match(/^Understanding/i)) {
      return cleanLine;
    }
  }
  
  return null;
};

// Helper function to remove "Thinking..." sections from report content
const removeThinkingSections = (content) => {
  if (!content) return content;
  
  // Check if content has "Thinking..." sections
  if (content.includes("Thinking...")) {
    // Find the first occurrence of a meaningful header (like "# Salesforce's Investment Thesis")
    // or the first non-thinking paragraph that appears to be part of the final report
    
    // Look for the first Markdown header that likely starts the actual report
    const headerMatch = content.match(/^#+\s+[^*\n]+/m);
    
    if (headerMatch && headerMatch.index > 0) {
      // Return content starting from the first header
      return content.substring(headerMatch.index);
    }
    
    // If we can't find a header, try to find where the actual report begins
    // by looking for patterns that indicate the end of thinking sections
    const sections = content.split(/\n\s*\n/); // Split by empty lines
    let startIndex = 0;
    
    // Find where thinking sections end and real content begins
    for (let i = 0; i < sections.length; i++) {
      const section = sections[i];
      // If the section doesn't start with "Thinking..." or "**", and appears substantive
      if (!section.includes("Thinking...") && !section.includes("**") && 
          section.length > 100 && !section.match(/^\s*\*\*/)) {
        startIndex = i;
        break;
      }
    }
    
    if (startIndex > 0) {
      return sections.slice(startIndex).join("\n\n");
    }
  }
  
  return content; // Return original if no thinking sections found
};

// Function to render web links in a structured format
const renderWebLinks = (links) => {
  console.log('[DetailsPanel - renderWebLinks] Received links:', links);
  if (!links || !Array.isArray(links) || links.length === 0) {
    return '<div class="text-gray-500 italic">No web links available</div>';
  }

  const generatedHtml = `
    <div class="space-y-3">
      ${links.map(link => {
        // Handle different link formats safely
        const url = typeof link === 'string' ? link : link.url || link.href || '';
        const title = link.title || link.name || (typeof link === 'string' ? link : url);
        const description = link.description || link.summary || link.snippet || '';
        
        // Safely extract domain for favicon
        let domain = 'unknown';
        let faviconUrl = '';
        try {
          if (url && url.includes('://')) {
            domain = new URL(url).hostname;
            faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
          } else if (url && url.includes('.')) {
            // Handle URLs without protocol
            domain = url.split('/')[0];
            faviconUrl = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
          }
        } catch (e) {
          console.warn('Error parsing URL for favicon:', e);
          faviconUrl = ''; // No favicon if URL parsing fails
        }
        
        // Only render if we have a valid URL
        if (!url) return '';
        
        return `
          <div class="border-b border-gray-100 pb-3 last:border-b-0">
            <div class="flex items-start">
              <div class="flex-shrink-0 mt-1">
                ${faviconUrl ? 
                  `<img src="${faviconUrl}" class="w-5 h-5" alt="${domain}" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" />
                   <svg class="w-5 h-5 text-blue-500 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                   </svg>` : 
                  `<svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                   </svg>`
                }
              </div>
              <div class="ml-3 flex-1">
                <a href="${url}" target="_blank" class="font-medium text-blue-700 hover:underline block">${title}</a>
                ${description ? `<div class="text-sm text-gray-600 mt-1">${description}</div>` : ''}
                <div class="text-xs text-gray-500 mt-1">${domain}</div>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
  console.log('[DetailsPanel - renderWebLinks] Generated HTML for web links (first 300 chars):', generatedHtml.substring(0, 300) + '...');
  return generatedHtml;
};

// Function to render code snippets with CodeSnippetViewer component
const renderCodeSnippets = (snippets) => {
  if (!snippets || !Array.isArray(snippets) || snippets.length === 0) {
    return '<div class="text-gray-500 italic">No code snippets available</div>';
  }

  // Return a placeholder div that will be replaced by React
  return `
    <div class="space-y-4" id="code-snippets-container" data-snippets='${JSON.stringify(snippets).replace(/'/g, "&#39;")}'>
      <div class="text-gray-500 italic">Loading code snippets...</div>
    </div>
  `;
};

// Function to render images
const renderImages = (images) => {
  if (!images || !Array.isArray(images) || images.length === 0) {
    return '<div class="text-gray-500 italic">No images available</div>';
  }

  return `
    <div class="space-y-4">
      ${images.map((image, index) => {
        return `
          <div class="border border-gray-200 rounded-md overflow-hidden">
            <img src="${image.src || image.url}" alt="${image.description || 'Research image'}" class="w-full h-auto" />
            ${image.description ? `<div class="p-3 text-sm text-gray-600">${image.description}</div>` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
};

// Enhanced function to generate more detailed HTML for various content types
const generateEnhancedDetailsHtml = (item) => {
  if (!item) return '';

  // Initialize HTML sections
  let contentHtml = '';
  let webLinksHtml = '';
  let codeSnippetsHtml = '';
  let imagesHtml = '';
  
  // Debug logging
  console.log('[DetailsPanel - generateEnhancedDetailsHtml] Processing item:', item);
  
  // Process main content
  if (item.content) {
    contentHtml = `
      <div class="mb-6">
        <h3 class="text-lg font-medium mb-3">Content</h3>
        <div class="prose max-w-none">
          ${item.content.replace(/`([^`]+)`/g, '<code class="inline-block px-2 py-1 bg-gray-100 border border-gray-300 rounded shadow-sm font-mono text-sm text-gray-800">$1</code>')}
        </div>
      </div>
    `;
  }
  
  // Process enriched data if available
  if (item.enrichedData) {
    console.log('Found enriched data:', item.enrichedData);
    console.log('[DetailsPanel - generateEnhancedDetailsHtml] item.enrichedData:', item.enrichedData);
    
    // Extract web links from various possible properties
    const extractLinks = () => {
      const allLinks = [];
      
      // Check common link properties
      if (item.enrichedData.sources && Array.isArray(item.enrichedData.sources)) {
        allLinks.push(...item.enrichedData.sources);
      }
      
      if (item.enrichedData.links && Array.isArray(item.enrichedData.links)) {
        allLinks.push(...item.enrichedData.links);
      }
      
      if (item.enrichedData.urls && Array.isArray(item.enrichedData.urls)) {
        allLinks.push(...item.enrichedData.urls);
      }
      
      // Check for domains field which may contain URLs
      if (item.enrichedData.domains && Array.isArray(item.enrichedData.domains)) {
        const domainLinks = item.enrichedData.domains.map(domain => {
          return typeof domain === 'string' ? 
            { url: domain.includes('://') ? domain : `https://${domain}`, title: domain } : 
            domain;
        });
        allLinks.push(...domainLinks);
      }
      
      // Check if we have a single source object
      if (item.enrichedData.source && typeof item.enrichedData.source === 'object') {
        allLinks.push(item.enrichedData.source);
      }
      
      // Check for references
      if (item.enrichedData.references && Array.isArray(item.enrichedData.references)) {
        allLinks.push(...item.enrichedData.references);
      }
      
      console.log('Extracted links:', allLinks);
      return allLinks;
    };
    
    const links = extractLinks();
    console.log('[DetailsPanel - generateEnhancedDetailsHtml] Extracted links for renderWebLinks:', links);
    
    // Web links
    if (links.length > 0) {
      webLinksHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Web Links</h3>
          ${renderWebLinks(links)}
        </div>
      `;
    }
    
    // Code snippets
    if (item.enrichedData.code_snippets && item.enrichedData.code_snippets.length > 0) {
      codeSnippetsHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Code Snippets</h3>
          ${renderCodeSnippets(item.enrichedData.code_snippets)}
        </div>
      `;
    }
    
    // Images
    if (item.enrichedData.images && item.enrichedData.images.length > 0) {
      imagesHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Images</h3>
          ${renderImages(item.enrichedData.images)}
        </div>
      `;
    }
  }
  
  // Special handling for nodeData
  if (item.nodeData) {
    console.log('Found node data:', item.nodeData);
    
    // Handle sources in nodeData
    if (item.nodeData.sources && Array.isArray(item.nodeData.sources) && !webLinksHtml) {
      webLinksHtml = `
        <div class="mb-6">
          <h3 class="text-lg font-medium mb-3">Web Links</h3>
          ${renderWebLinks(item.nodeData.sources)}
        </div>
      `;
    }
  }
  
  // Combine all HTML sections
  return `
    <div class="details-content">
      ${contentHtml}
      ${webLinksHtml}
      ${codeSnippetsHtml}
      ${imagesHtml}
    </div>
  `;
};

function DetailsPanel({ 
  isVisible, 
  onClose, 
  selectedItem, 
  showFinalReport, 
  reportContent,
  query,
  isResearching
}) {
  const [progress, setProgress] = useState(0);
  const [isLive] = useState(true);
  const [reportTitle, setReportTitle] = useState(null);
  const [isFocusedView, setIsFocusedView] = useState(false);
  const [filteredReportContent, setFilteredReportContent] = useState(null);
  const [isRendered, setIsRendered] = useState(false);

  // Handle animation states
  useEffect(() => {
    if (isVisible) {
      // Small delay to ensure DOM is ready before triggering animations
      setTimeout(() => {
        setIsRendered(true);
      }, 50);
    } else {
      setIsRendered(false);
    }
  }, [isVisible]);

  // Update progress as research continues and filter report content
  useEffect(() => {
    if (!reportContent) {
      setProgress(0);
      setReportTitle(null);
      setFilteredReportContent(null);
      // Reset document title
      document.title = 'AI Research Assistant';
    } else {
      setProgress(100);
      
      // Filter out thinking sections from report content
      const cleanedContent = removeThinkingSections(reportContent);
      setFilteredReportContent(cleanedContent);
      
      // Extract title from filtered report content
      const title = extractReportTitle(cleanedContent);
      setReportTitle(title);
      
      // Update document title with the report title
      if (title) {
        document.title = title;
      } else {
        // Fallback to query if no title was extracted
        document.title = query ? `Research: ${query}` : 'AI Research Assistant';
      }
    }
  }, [reportContent, query]);

  useEffect(() => {
    // Insert code snippets after the component has rendered
    if (isVisible && selectedItem && selectedItem.enrichedData && selectedItem.enrichedData.code_snippets) {
      const container = document.getElementById('code-snippets-container');
      if (container) {
        try {
          const snippets = JSON.parse(container.getAttribute('data-snippets'));
          
          // Clear the container first
          while (container.firstChild) {
            container.removeChild(container.firstChild);
          }
          
          // Store references to roots for cleanup
          const roots = [];
          
          // Render each snippet as a React component
          snippets.forEach(snippet => {
            const snippetContainer = document.createElement('div');
            snippetContainer.className = 'mb-4';
            container.appendChild(snippetContainer);
            
            // Use ReactDOM createRoot (React 18+)
            const root = ReactDOM.createRoot(snippetContainer);
            root.render(<CodeSnippetViewer snippet={snippet} initialCollapsed={false} />);
            roots.push(root);
          });
          
          // Cleanup function to unmount components when dependencies change
          return () => {
            roots.forEach(root => {
              try {
                root.unmount();
              } catch (e) {
                console.error('Error unmounting root:', e);
              }
            });
          };
        } catch (error) {
          console.error('Error rendering code snippets:', error);
        }
      }
    }
  }, [isVisible, selectedItem]);

  const handleFullscreen = () => {
    // Toggle focused view
    setIsFocusedView(!isFocusedView);
  };

  // Determine title based on content type
  let title = "Details";
  if (showFinalReport) {
    title = reportTitle || "Research Summary";
  } else if (selectedItem && selectedItem.title) {
    title = selectedItem.title;
  } else if (selectedItem) {
    // Fallback for items without a specific title, e.g. raw activity
    title = selectedItem.type ? selectedItem.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : "Details";
  }

  // Render content placeholder when no report is available
  const renderPlaceholder = () => (
    <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-8">
      <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>

      <p className="text-center max-w-md">
        {isResearching 
          ? "Research in progress. The report will appear here when ready." 
          : "Start a research query to generate a report."}
      </p>
    </div>
  );

  return (
    <div 
      id="details-panel" 
      className={`bg-white border border-gray-200 rounded-lg m-3 shadow-lg flex flex-col ${isVisible ? 'visible' : ''}`}
      aria-hidden={!isVisible}
    >
      {/* Fixed Header */}
      <div className="border-b border-gray-200 p-3 flex items-center justify-between bg-white sticky top-0 z-10 flex-shrink-0 rounded-t-lg"> 
        <div className="flex items-center space-x-3 flex-1">
          <div className="p-2 bg-gray-100 rounded flex-shrink-0">
            {showFinalReport ? (
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="font-medium text-lg truncate" id="details-title">
              {showFinalReport && query ? query : title}
            </h2>
          </div>
        </div>
        <div className="flex items-center ml-3">
          <button 
            id="minimize-details" 
            className="p-1 text-gray-500 hover:bg-gray-100 rounded flex items-center justify-center w-8 h-8"
            onClick={onClose}
            aria-label="Minimize panel"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto">
        {/* Content based on what's being shown */}
        {!showFinalReport && selectedItem && (
          <div 
            id="details-content"
            className="p-6" 
            dangerouslySetInnerHTML={{ __html: generateEnhancedDetailsHtml(selectedItem) }}
          />
        )}

        {/* Final Report */}
        {showFinalReport && (
          <div className="h-full">
            {filteredReportContent ? (
              <FinalReport reportContent={filteredReportContent} isFocusedView={isFocusedView} />
            ) : (
              renderPlaceholder()
            )}
          </div>
        )}
      </div>

      {/* Fixed Footer - Always show when panel is visible */}
      {isVisible && (
        <div className="border-t border-gray-200 p-3 flex items-center justify-between bg-white sticky bottom-0 z-10 flex-shrink-0 rounded-b-lg">
          <div className="flex items-center space-x-3">
            <button className="p-1 text-gray-500 hover:bg-gray-100 rounded flex items-center justify-center w-8 h-8">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <button className="p-1 text-gray-500 hover:bg-gray-100 rounded flex items-center justify-center w-8 h-8">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <div className="h-1 bg-gray-200 rounded-full w-48 overflow-hidden">
              <div 
                className="h-1 bg-blue-500 rounded-full transition-all duration-300" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
          <div className="flex items-center ml-3">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-1">
                {isLive && (
                  <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-1"></span>
                )}
                <span className="text-sm text-gray-500">
                  {isLive ? 'live' : 'static'}
                </span>
              </div>
              <div className="text-xs text-gray-400">
                v0.6.5
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DetailsPanel; 