import React from 'react';
import CodeSnippetViewer from './CodeSnippetViewer';
import CodeWithVisualization from './CodeWithVisualization';

// Helper function to safely extract hostname from URL
const getHostname = (url) => {
  try {
    // Check if the URL is valid and has a protocol
    if (!url) return null;
    
    // Add https:// if missing
    const urlWithProtocol = url.startsWith('http') ? url : `https://${url}`;
    const urlObj = new URL(urlWithProtocol);
    return urlObj.hostname;
  } catch (e) {
    console.error('Invalid URL:', url, e);
    return null;
  }
};

function ActivityItem({ activity, timestamp, enrichedData, itemType = 'default', index = 0 }) {
  
  // Helper function to render sources with proper formatting
  const renderSources = () => {
    if (!enrichedData) {
      return null;
    }

    // Check if we have sources data
    const hasSources =
      enrichedData.sources || (enrichedData.source_titles && enrichedData.source_urls) || enrichedData.source_count

    if (!hasSources) {
      return null;
    }
    
    // Helper function to render a single source item
    const renderSourceItem = (source, index) => {
      const url = source.url || source;
      const title = source.title || '';
      const hostname = getHostname(url);
      const faviconUrl = hostname ? `https://www.google.com/s2/favicons?sz=32&domain=${hostname}` : null;
      const displayText = (title || url || '').replace(/^\*\s*/, '');
      
      return (
        <div key={index} className="flex items-center space-x-2 p-1 bg-gray-100 hover:bg-gray-200 rounded-md">
          <div className="w-5 h-5 flex-shrink-0 bg-gray-200 rounded-full overflow-hidden flex items-center justify-center text-xs text-gray-500">
            {faviconUrl ? (
              <img
                src={faviconUrl}
                alt=""
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.currentTarget.src = "";
                  e.currentTarget.parentElement.textContent = title?.[0] || "S";
                }}
              />
            ) : (
              title?.[0] || "S"
            )}
          </div>
          {url ? (
            <a
              href={url.startsWith('http') ? url : `https://${url}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-gray-700 hover:text-blue-600 truncate"
              title={displayText}
            >
              {displayText}
            </a>
          ) : (
            <span className="text-xs text-gray-700 truncate" title={displayText}>
              {displayText}
            </span>
          )}
        </div>
      );
    };

    return (
      <div className="source-section mt-3 pl-4">
        {/* Display combined sources if available */}
        {enrichedData.sources && Array.isArray(enrichedData.sources) && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {enrichedData.sources.map((source, index) => renderSourceItem(source, index))}
          </div>
        )}

        {/* Display separate title/url lists if available */}
        {!enrichedData.sources && enrichedData.source_titles && Array.isArray(enrichedData.source_titles) && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {enrichedData.source_titles.map((title, index) => {
              const url = enrichedData.source_urls && enrichedData.source_urls[index];
              return renderSourceItem({ title, url }, index);
            })}
          </div>
        )}

        {/* If we only have a count but no actual sources listed */}
        {enrichedData.source_count &&
          !enrichedData.sources &&
          (!enrichedData.source_titles || !enrichedData.source_titles.length) && (
            <div className="text-sm text-gray-600">{enrichedData.source_count} sources identified</div>
          )}
      </div>
    )
  }

  // Helper to render domains if available
  const renderDomains = () => {
    if (!enrichedData || !enrichedData.domains || enrichedData.domains.length === 0) {
      return null
    }

    // Categorize domains
    const categorizedDomains = {
      Academic: [],
      Professional: [],
      Other: [],
    }

    enrichedData.domains.forEach((domain) => {
      if (!domain || typeof domain.url !== "string") return
      const url = domain.url.toLowerCase()
      if (
        url.includes(".edu") ||
        url.includes(".org") ||
        url.includes("ieee.org") ||
        url.includes("arxiv.org") ||
        url.includes("aclweb.org") ||
        url.includes("msrconf.org")
      ) {
        categorizedDomains.Academic.push(domain)
      } else if (url.includes("linkedin.com")) {
        categorizedDomains.Professional.push(domain)
      } else {
        categorizedDomains.Other.push(domain)
      }
    })

    // Only render if there are domains to show
    if (
      categorizedDomains.Academic.length === 0 &&
      categorizedDomains.Professional.length === 0 &&
      categorizedDomains.Other.length === 0
    ) {
      return null
    }

    // Helper function to get favicon URL
    const getFaviconUrl = (url) => {
      try {
        const domain = new URL(url).hostname
        return `https://www.google.com/s2/favicons?sz=32&domain_url=${domain}`
      } catch (e) {
        return ""
      }
    }

    return (
      <div className="domains-section mt-4">
        <div className="text-sm font-medium text-gray-800 mb-2 pl-4">Search, Read and Browsing</div>
        <div className="grid grid-cols-3 gap-2 pl-4">
          {enrichedData.domains.map((domain, index) => {
            const faviconUrl = getFaviconUrl(domain.url)
            return (
              <div key={index} className="flex items-center space-x-2 p-1 bg-gray-100 hover:bg-gray-200 rounded-md">
                <div className="w-5 h-5 flex-shrink-0 bg-gray-200 rounded-full overflow-hidden flex items-center justify-center text-xs text-gray-500">
                  {faviconUrl ? (
                    <img
                      src={faviconUrl || "/placeholder.svg"}
                      alt=""
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.currentTarget.src = ""
                        e.currentTarget.parentElement.textContent = domain.title?.[0] || "D"
                      }}
                    />
                  ) : (
                    domain.title?.[0] || "D"
                  )}
                </div>
                <a
                  href={domain.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-gray-700 hover:text-blue-600 truncate"
                  title={domain.url}
                >
                  {(domain.title || new URL(domain.url).hostname)?.replace(/^\*\s*/, '')}
                </a>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Helper to render subtopics if available
  const renderSubtopics = () => {
    if (!enrichedData || !enrichedData.subtopics || !Array.isArray(enrichedData.subtopics)) {
      return null
    }

    return (
      <div className="subtopics-section mt-3 pl-4">
        <div className="text-sm font-medium text-gray-800 mb-2">Future Research Directions</div>
        <div className="text-sm text-gray-700 pl-0">
          {enrichedData.subtopics.map((subtopic, index) => (
            <p key={index} className="mb-2">
              {subtopic}
            </p>
          ))}
        </div>
      </div>
    )
  }

  // Helper to render knowledge gaps if available
  const renderGaps = () => {
    if (!enrichedData || (!enrichedData.gaps && !enrichedData.gap_list)) {
      return null
    }

    return (
      <div className="gaps-section mt-3 pl-4">
        <div className="text-sm font-medium text-gray-800 mb-2">Knowledge gaps identified:</div>
        {Array.isArray(enrichedData.gap_list) ? (
          <div className="text-sm text-gray-700">
            {enrichedData.gap_list.map((gap, index) => (
              <p key={index} className="mb-2">
                {gap}
              </p>
            ))}
          </div>
        ) : (
          <div className="text-sm text-gray-700">{enrichedData.gaps || "Areas requiring more research"}</div>
        )}
      </div>
    )
  }

  // Debug: log visualization images
  console.debug('ActivityItem details:', {
    activity: activity,
    hasEnrichedData: !!enrichedData,
    hasImages: enrichedData && 'images' in enrichedData,
    images: enrichedData?.images,
    imageCount: enrichedData?.images?.length,
    relatedEventType: enrichedData?.related_event_type,
    nodeName: enrichedData?.node_name,
  });
  
  // Combined method to render code snippets with their associated visualizations
  const renderCodeAndVisualizations = () => {
    console.debug('[ActivityItem] renderCodeAndVisualizations called');
    
    // Track processed images to avoid duplicates
    const processedImageSrcs = new Set();
    
    // Collection of code snippets that have associated visualizations
    const codeWithVisuals = [];
    
    // Collection of standalone code snippets without visualizations
    const standaloneCodeSnippets = [];
    
    // Collection of standalone images without associated code
    const standaloneImages = [];
    
    // First, process code snippets and check if they have visualizations
    if (enrichedData?.code_snippets && Array.isArray(enrichedData.code_snippets)) {
      const validSnippets = enrichedData.code_snippets.filter(snippet => 
        snippet && typeof snippet === 'object' && snippet.code
      );
      
      validSnippets.forEach(snippet => {
        if (snippet.visualization) {
          // This is a code snippet with an associated visualization
          codeWithVisuals.push(snippet);
          
          // Mark this visualization as processed
          const imageSrc = snippet.visualization.src || 
            (snippet.visualization.data ? 
              `data:image/${snippet.visualization.format || 'png'};base64,${snippet.visualization.data}` : 
              null);
          
          if (imageSrc) {
            processedImageSrcs.add(imageSrc);
          }
        } else {
          // This is a standalone code snippet
          standaloneCodeSnippets.push(snippet);
        }
      });
    }
    
    // Then, process any standalone images not associated with code
    if (enrichedData?.images && Array.isArray(enrichedData.images)) {
      enrichedData.images.forEach(img => {
        const imageSrc = img.src || (img.data ? 
          `data:image/${img.format || 'png'};base64,${img.data}` : 
          null);
        
        // Skip if the image source is invalid or already processed
        if (!imageSrc || processedImageSrcs.has(imageSrc)) {
          return;
        }
        
        // This is a standalone image
        standaloneImages.push(img);
      });
    }
    
    // Debugging
    console.debug(`[ActivityItem] Found ${codeWithVisuals.length} code snippets with visualizations`);
    console.debug(`[ActivityItem] Found ${standaloneCodeSnippets.length} standalone code snippets`);
    console.debug(`[ActivityItem] Found ${standaloneImages.length} standalone images`);
    
    // Return null if there's nothing to render
    if (codeWithVisuals.length === 0 && standaloneCodeSnippets.length === 0 && standaloneImages.length === 0) {
      return null;
    }
    
    return (
      <>
        {/* Render code snippets with associated visualizations */}
        {codeWithVisuals.length > 0 && (
          <div className="code-vis-section mt-3 pl-4">
            <div className="text-sm font-medium text-gray-800 mb-2">Generating, Analyzing, and Visualizing Data</div>
            {codeWithVisuals.map((snippet, index) => (
              <CodeWithVisualization key={`code-vis-${index}`} snippet={snippet} />
            ))}
          </div>
        )}
        
        {/* Render standalone code snippets */}
        {standaloneCodeSnippets.length > 0 && (
          <div className="code-snippets-section mt-3 pl-4">
            <div className="text-sm font-medium text-gray-800 mb-2">Generating, Analyzing, and Optimizing Code</div>
            {standaloneCodeSnippets.map((snippet, index) => (
              <CodeSnippetViewer key={`code-${index}`} snippet={snippet} />
            ))}
          </div>
        )}
        
        {/* Render standalone images */}
        {standaloneImages.length > 0 && (
          <div className="images-section mt-3 pl-4">
            <div className="text-sm font-medium text-gray-800 mb-2">Research Visualization</div>
            {standaloneImages.map((img, index) => {
              const imageSrc = img.src || (img.data ? 
                `data:image/${img.format || 'png'};base64,${img.data}` : 
                null);
                
              if (!imageSrc) return null;
              
              return (
                <img
                  key={`img-${index}`}
                  src={imageSrc}
                  alt={img.description || "Research Visualization"}
                  title={img.description}
                  loading="lazy"
                  className="rounded shadow-sm transition-opacity duration-300 mb-2 max-w-full h-auto"
                  onError={(e) => {
                    console.error(`[ActivityItem] Error loading standalone image ${index}:`, e);
                    e.target.style.display = 'none';
                  }}
                />
              );
            })}
          </div>
        )}
      </>
    );
  };
  
  // Legacy rendering functions - commented out as they're replaced by renderCodeAndVisualizations
  // These are kept for reference and possible rollback if needed
  /*
  const renderVisualizations = () => {
    // Function implementation...
  };

  const renderCodeSnippets = () => {
    // Function implementation...
  };
  */

  // Diamond icon helper (copied from ResearchItemList.js for consistency)
  const getItemIcon = (type) => {
    return (
      <svg width="12" height="12" viewBox="0 0 12 12" className="transform rotate-45">
        <rect x="3" y="3" width="6" height="6" fill="#000" opacity="0.9" />
      </svg>
    );
  };

  return (
    <div className="activity-item mb-4">
      <div className="flex items-start">
        <div className="item-number mr-3" style={{marginTop: '-5px'}}>{getItemIcon(itemType) || (index + 1)}</div>
        <div className="flex-1 py-1" style={{marginLeft: '34px'}} >
          <div className="text-base font-medium text-gray-800 mb-1">{activity}</div>
          <div className="text-sm text-gray-700 mb-2">{enrichedData?.description || ""}</div>

          {/* Render specialized sections for common enriched data types */}
          {renderSubtopics()}
          {renderGaps()}
          {renderDomains()}
          {renderSources()}
          {renderCodeAndVisualizations()}
          {/* The following methods are kept for backward compatibility */}
          {/* {renderVisualizations()} */}
          {/* {renderCodeSnippets()} */}

          {/* For any other enriched data not covered by specialized renderers */}
          {enrichedData && Object.keys(enrichedData).length > 0 && (
            <div className="text-sm text-gray-600 pl-4 mt-3">
              {Object.entries(enrichedData)
                .filter(
                  ([key]) =>
                    ![
                      "sources",
                      "source_titles",
                      "source_urls",
                      "source_count",
                      "domains",
                      "domain_count",
                      "subtopics",
                      "gaps",
                      "gap_list",
                      "description",
                      "images",
                      "code_snippets", // Add code_snippets here
                    ].includes(key),
                )
                .map(([key, value]) => {
                  // Skip rendering complex objects or arrays that we don't have special handling for
                  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
                    return null
                  }

                  return (
                    <div key={key} className="flex items-start mb-1">
                      <span className="font-medium mr-1">{key.replace(/_/g, " ")}:</span>
                      <span>
                        {typeof value === "string"
                          ? value
                          : Array.isArray(value) && value.length <= 3
                            ? value.join(", ")
                            : Array.isArray(value)
                              ? `${value.length} items`
                              : JSON.stringify(value)}
                      </span>
                    </div>
                  )
                })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ActivityItem