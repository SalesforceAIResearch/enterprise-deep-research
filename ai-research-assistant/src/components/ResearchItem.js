import React from 'react';
import ActivityItem from './ActivityItem';

function ResearchItem({ item, isActive, onClick }) { // isActive might not be needed now, onClick passed down
  const { id, title, timestamp, content, activityText, enrichedData, type } = item;

  // Determine the main activity text for this item
  // Prioritize activityText, then title, then content
  const mainActivityText = activityText || title || content || 'Processing...'; 

  // Accept index as a prop if passed from parent (default 0)
  // Accept itemType as item.type or 'default'
  const itemType = type || 'default';
  const itemIndex = typeof item.index === 'number' ? item.index : 0;

  // Always render using ActivityItem structure for consistency
  return (
    <div onClick={onClick} className={`research-item-wrapper ${isActive ? 'active' : ''}`}> 
      <ActivityItem 
        activity={mainActivityText}
        timestamp={timestamp}
        enrichedData={enrichedData}
        itemType={itemType}
        index={itemIndex}
      />
    </div>
  );
}

export default ResearchItem;