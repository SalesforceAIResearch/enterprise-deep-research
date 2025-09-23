/**
 * Helper utilities shared across components
 */

/**
 * Generate detailed HTML for a research item
 * @param {Object} item - The research item object
 * @returns {string} HTML string for the item details
 */
export const generateDetailsHtml = (item) => {
  if (!item) return '';
  
  let html = `<div class="p-4">
    <h3 class="font-semibold mb-3">${item.title}</h3>
    <div class="mb-3">${item.content}</div>`;
  
  // Add node data if available
  if (item.nodeData) {
    // Format based on node type
    if (item.title === "Sources Found" && item.nodeData.sources) {
      html += `<div class="bg-gray-100 p-3 rounded-md mb-4">
        <h4 class="font-medium mb-2">Sources:</h4>
        <ul class="space-y-2">
          ${item.nodeData.sources.map(source => `
            <li class="flex items-center">
              <img src="https://www.google.com/s2/favicons?domain=${new URL(source.url).hostname}&sz=32" class="w-4 h-4 mr-2" />
              <a href="${source.url}" target="_blank" class="text-blue-600 hover:underline">${source.title || source.url}</a>
            </li>
          `).join('')}
        </ul>
      </div>`;
    } else if (item.nodeData.sources_gathered) {
      html += `<div class="bg-gray-100 p-3 rounded-md mb-4">
        <h4 class="font-medium mb-2">Sources Retrieved:</h4>
        <ul class="list-disc pl-5 space-y-1">
          ${item.nodeData.sources_gathered.map(source => `<li>${source}</li>`).join('')}
        </ul>
      </div>`;
      
      if (item.nodeData.web_research_results && item.nodeData.web_research_results.length > 0) {
        html += `<h4 class="font-medium mb-2">Source Content:</h4>
        <div class="border-l-2 border-gray-400 pl-3 py-1 mb-3 text-sm">
          ${item.nodeData.web_research_results[0].replace(/\n/g, '<br>')}
        </div>`;
      }
    } else if (item.title === "Searching for Information" && item.nodeData.input) {
      html += `<div class="bg-gray-100 p-3 rounded-md mb-4">
        <h4 class="font-medium mb-2">Search Details:</h4>
        <p><strong>Query:</strong> "${item.nodeData.input.search_query || 'Not specified'}"</p>
        ${item.nodeData.input.subtopic_queries && item.nodeData.input.subtopic_queries.length > 0 ? 
          `<p class="mt-2"><strong>Subtopics:</strong></p>
          <ul class="list-disc pl-5">
            ${item.nodeData.input.subtopic_queries.map(q => `<li>${q}</li>`).join('')}
          </ul>` : ''}
      </div>`;
    } else if (item.title === "Planning Research Approach" && item.nodeData.output) {
      html += `<div class="bg-gray-100 p-3 rounded-md mb-4">
        <h4 class="font-medium mb-2">Research Plan:</h4>
        <div class="whitespace-pre-line text-sm">
          ${item.nodeData.output.replace(/\n/g, '<br>')}
        </div>
      </div>`;
    } else if (item.nodeData.status) {
      html += `<div class="bg-gray-100 p-3 rounded-md mb-4">
        <h4 class="font-medium mb-2">Status:</h4>
        <p>${item.nodeData.status}</p>
      </div>`;
    }
  }
  
  // Add enriched data if available
  if (item.enrichedData) {
    // Add images if available
    if (item.enrichedData.images && item.enrichedData.images.length > 0) {
      html += `<div class="mb-4">
        <h4 class="font-medium mb-2">Images:</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          ${item.enrichedData.images.map(image => `
            <div class="border border-gray-200 rounded overflow-hidden">
              <img src="${image.src || image.url}" alt="${image.description || 'Research image'}" class="w-full h-auto" />
              ${image.description ? `<div class="p-2 text-sm text-gray-600">${image.description}</div>` : ''}
            </div>
          `).join('')}
        </div>
      </div>`;
    }
    
    // Add code snippets if available
    if (item.enrichedData.code_snippets && item.enrichedData.code_snippets.length > 0) {
      html += `<div class="mb-4">
        <h4 class="font-medium mb-2">Code Snippets:</h4>
        <div class="space-y-3">
          ${item.enrichedData.code_snippets.map(snippet => `
            <div class="rounded overflow-hidden border border-gray-200">
              <div class="bg-gray-100 px-3 py-1 text-sm flex justify-between">
                <span class="font-mono">${snippet.language || 'code'}</span>
                <button onclick="navigator.clipboard.writeText(\`${snippet.code.replace(/`/g, '\\`')}\`)" class="text-blue-600 hover:text-blue-800">Copy</button>
              </div>
              <pre class="bg-gray-900 text-gray-200 p-3 overflow-x-auto text-sm"><code>${snippet.code}</code></pre>
            </div>
          `).join('')}
        </div>
      </div>`;
    }
    
    // Add sources/links if available
    if (item.enrichedData.sources || item.enrichedData.links || item.enrichedData.urls || item.enrichedData.domains) {
      // Extract all possible links from various properties
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
        
        return allLinks;
      };
      
      const links = extractLinks();
      
      if (links.length > 0) {
        html += `<div class="mb-4">
          <h4 class="font-medium mb-2">Sources:</h4>
          <div class="space-y-2">
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
                faviconUrl = ''; // No favicon if URL parsing fails
              }
              
              // Only render if we have a valid URL
              if (!url) return '';
              
              return `
                <div class="web-link-item">
                  <div class="flex items-start">
                    <div class="flex-shrink-0 mt-1">
                      ${faviconUrl ? 
                        `<img src="${faviconUrl}" class="web-link-icon" alt="${domain}" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" />
                         <svg class="web-link-icon text-blue-500 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                         </svg>` : 
                        `<svg class="web-link-icon text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                         </svg>`
                      }
                    </div>
                    <div class="ml-3 flex-1">
                      <a href="${url}" target="_blank" class="web-link-title">${title}</a>
                      ${description ? `<div class="web-link-description">${description}</div>` : ''}
                      <div class="web-link-domain">${domain}</div>
                    </div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>`;
      }
    }
  }
  
  html += `</div>`;
  return html;
}; 