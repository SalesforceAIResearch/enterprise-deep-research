"use client"
import ResearchItem from "./ResearchItem"

// Modern CSS with contemporary design principles
const connectorStyles = `
  .connector-line-vertical {
    position: absolute;
    left: 12px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%);
    border-radius: 1px;
  }
  
  .timeline-continuous-line {
    position: absolute;
    left: 12px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
    border-radius: 1px;
    z-index: 1;
  }
  
  .task-item {
    position: relative;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    padding: 1rem 1.25rem 1rem 2.5rem;
    margin-bottom: 0.5rem;
    border-radius: 12px;
    border: 1px solid transparent;
    background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
  }
  
  .task-item.assistant-manus-intro {
    padding-left: 1.25rem;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
  }
  
  .task-item:hover {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-color: #cbd5e1;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.02);
  }
  
  .task-item.selected {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-color: #3b82f6;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1), 0 2px 8px rgba(59, 130, 246, 0.05);
  }
  
  .source-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
    margin-top: 0.5rem;
  }
  
  .source-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0;
  }
  
  .task-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  .timestamp {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.025em;
  }
  
  .search-query {
    font-size: 0.875rem;
    color: #475569;
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }
  
  .reading-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 0.25rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  
  .item-number {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
    position: absolute;
    left: 0;
    top: 1rem;
    z-index: 10;
    font-weight: 600;
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  .task-item:hover .item-number {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-color: #cbd5e1;
    transform: scale(1.05);
  }
  
  .task-item.selected .item-number {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    border-color: #1d4ed8;
    color: white;
    transform: scale(1.1);
  }
  
  .type-icon {
    margin-right: 0.75rem;
    flex-shrink: 0;
  }
  
  .light-bg {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  }
  
  .item-content {
    margin-top: 0.5rem;
    line-height: 1.6;
    color: #334155;
    font-weight: 400;
  }
  
  .task-item.with-children {
    margin-bottom: 0.25rem;
    padding-bottom: 0.75rem;
  }
  
  .children-container {
    position: relative;
    margin-left: 1.5rem;
    padding-top: 0.5rem;
  }
  
  /* Modern typography improvements */
  .task-item {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
  }
  
  /* Enhanced focus states for accessibility */
  .task-item:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }
  
  /* Subtle animation for the timeline */
  .timeline-continuous-line {
    animation: subtle-pulse 4s ease-in-out infinite;
  }
  
  @keyframes subtle-pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 0.8; }
  }
  
  /* Modern scrollbar styling */
  .research-timeline::-webkit-scrollbar {
    width: 6px;
  }
  
  .research-timeline::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }
  
  .research-timeline::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
  }
  
  .research-timeline::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
`

function ResearchItemList({ items, selectedItemId, onItemClick }) {
  if (!items || items.length === 0) {
    return null
  }

  // Modern icon design with subtle styling
  const getItemIcon = (type) => {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" className="transform rotate-45">
        <rect x="2" y="2" width="6" height="6" fill="currentColor" opacity="0.7" rx="1" />
      </svg>
    )
  }

  const renderItem = (item, index, depth = 0, isChild = false, isLastChild = false) => {
    const indentClass = depth > 0 ? `ml-${depth * 4}` : ""
    const hasChildren = item.children && item.children.length > 0
    const isSelected = selectedItemId === item.id
    const itemType = item.type || "default"

    if (itemType === "thought-process") {
      return (
        <div key={item.id} className={`relative ${indentClass}`}>
          <ResearchItem item={item} isActive={isSelected} onClick={() => onItemClick(item)} />
        </div>
      )
    }

    if (itemType === "assistant-manus-intro") {
      return (
        <div
          key={item.id}
          className={`task-item ${isSelected ? "selected" : ""} ${itemType} cursor-pointer`}
          onClick={() => onItemClick(item)}
        >
          <div className="flex items-center mb-3">
            <div className="mr-3 text-slate-600">{getItemIcon(itemType)}</div>
            <div className="font-semibold text-slate-800 text-base">{item.title}</div>
            <div className="timestamp ml-auto">{item.timestamp}</div>
          </div>
          <div className="text-sm text-slate-600 leading-relaxed" dangerouslySetInnerHTML={{ __html: item.content }} />
        </div>
      )
    }

    // Special handling for final-report-link to remove timeline elements
    if (itemType === "final-report-link") {
      return (
        <div key={item.id} className="relative">
          <div
            className={`${isSelected ? "selected" : ""} ${itemType} cursor-pointer`}
            style={{ 
              padding: '1rem 1.25rem', 
              marginBottom: '0.5rem', 
              borderRadius: '12px', 
              border: '1px solid transparent', 
              background: 'linear-gradient(135deg, #ffffff 0%, #fafbfc 100%)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
            }}
            onClick={() => onItemClick(item)}
          >
            <div className="item-content text-sm" dangerouslySetInnerHTML={{ __html: item.content }} />
          </div>
        </div>
      )
    }

    return (
      <div key={item.id} className="relative">
        <div
          className={`task-item ${isSelected ? "selected" : ""} ${itemType} ${hasChildren ? "with-children" : ""} cursor-pointer`}
          onClick={() => onItemClick(item)}
        >
          <div className={`item-number`}>{getItemIcon(itemType)}</div>

          <div className="item-content text-sm" dangerouslySetInnerHTML={{ __html: item.content }} />
        </div>

        {hasChildren && (
          <div className="children-container">
            <div className="connector-line-vertical" />
            {item.children.map((child, childIndex) =>
              renderItem(child, childIndex, depth + 1, true, childIndex === item.children.length - 1),
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="research-timeline relative mb-8">
      <style dangerouslySetInnerHTML={{ __html: connectorStyles }} />
      {/* Only show timeline line if there are non-final-report-link items */}
      {items.length > 0 && items.some(item => item.type !== "final-report-link") && (
        <div 
          className="timeline-continuous-line" 
          style={{
            // Calculate height to stop before final-report-link items
            height: (() => {
              const lastNonFinalReportIndex = items.map((item, index) => 
                item.type !== "final-report-link" ? index : -1
              ).filter(index => index !== -1).pop();
              
              if (lastNonFinalReportIndex !== undefined && lastNonFinalReportIndex < items.length - 1) {
                // Stop the line after the last non-final-report item
                return `calc(${(lastNonFinalReportIndex + 1) * 100}% / ${items.length} - 1rem)`;
              }
              return '100%';
            })()
          }}
        />
      )}
      {items.map((item, index) => renderItem(item, index))}
    </div>
  )
}

export default ResearchItemList
