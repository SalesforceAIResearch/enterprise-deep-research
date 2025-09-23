import React, { useState, useCallback } from 'react';
import ResearchPanel from './components/ResearchPanel';
import DetailsPanel from './components/DetailsPanel'; // Now handles both item details and report content
import './App.css'; // For global styles

function App() {
  const [isResearching, setIsResearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');
  const [extraEffort, setExtraEffort] = useState(false);
  const [minimumEffort, setMinimumEffort] = useState(false);
  const [benchmarkMode, setBenchmarkMode] = useState(false);
  const [modelProvider, setModelProvider] = useState('google'); // Default provider
  const [modelName, setModelName] = useState('gemini-2.5-pro-preview-03-25'); // Default model
  const [uploadedFileContent, setUploadedFileContent] = useState(null); // Added state for uploaded file content

  const [isDetailsPanelOpen, setIsDetailsPanelOpen] = useState(false);
  const [detailsPanelContentType, setDetailsPanelContentType] = useState(null); // 'item' or 'report'
  const [detailsPanelContentData, setDetailsPanelContentData] = useState(null);

  const handleBeginResearch = useCallback((query, extra, minimum, benchmark, modelConfig, fileContent) => { // Added fileContent
    setCurrentQuery(query);
    setExtraEffort(extra);
    setMinimumEffort(minimum);
    setBenchmarkMode(benchmark);
    if (modelConfig) {
      setModelProvider(modelConfig.provider);
      setModelName(modelConfig.model);
    }
    
    setUploadedFileContent(fileContent); // Set uploaded file content
    setIsResearching(true);
    setIsDetailsPanelOpen(false); // Close details panel when new research starts
    // Wait for animation to complete before resetting content
    setTimeout(() => {
      setDetailsPanelContentType(null);
      setDetailsPanelContentData(null);
    }, 300);
  }, [uploadedFileContent]);

  const handleShowItemDetails = useCallback((item) => {
    // Set data first, then trigger animation
    setDetailsPanelContentData(item);
    setDetailsPanelContentType('item');
    // Small delay to ensure data is set before animation starts
    setTimeout(() => {
      setIsDetailsPanelOpen(true);
    }, 10);
  }, []);

  const handleShowReportDetails = useCallback((reportContent) => {
    // Set data first, then trigger animation
    setDetailsPanelContentData(reportContent);
    setDetailsPanelContentType('report');
    // Small delay to ensure data is set before animation starts
    setTimeout(() => {
      setIsDetailsPanelOpen(true);
    }, 10);
  }, []);

  const handleCloseDetailsPanel = useCallback(() => {
    setIsDetailsPanelOpen(false);
    // Wait for animation to complete before clearing data
    setTimeout(() => {
      setDetailsPanelContentData(null);
      setDetailsPanelContentType(null);
    }, 300); // Match the transition timing in CSS
  }, []);
  
  // This callback is used by ResearchPanel to inform App.js that a report is ready.
  const [finalReportData, setFinalReportData] = useState(null);
  const handleReportGenerated = useCallback((report) => {
    setFinalReportData(report); // Store report data
    // Optionally, automatically open the report:
    // handleShowReportDetails(report);
  }, []);

  const handleStopResearch = useCallback(() => {
    console.log('Stopping research from App.js');
    setIsResearching(false);
  }, []);

  return (
    <div className={`app-container ${isDetailsPanelOpen ? 'details-panel-active' : ''}`}>
      <div className="main-panel-wrapper">
        <ResearchPanel 
          query={currentQuery}
          extraEffort={extraEffort}
          minimumEffort={minimumEffort}
          benchmarkMode={benchmarkMode}
          modelProvider={modelProvider}
          modelName={modelName}
          uploadedFileContent={uploadedFileContent} // Pass uploadedFileContent
          isResearching={isResearching}
          onBeginResearch={handleBeginResearch}
          onReportGenerated={handleReportGenerated}
          onShowItemDetails={handleShowItemDetails}
          onShowReportDetails={handleShowReportDetails}
          onStopResearch={handleStopResearch}
        />
      </div>
      
      <DetailsPanel
        isVisible={isDetailsPanelOpen}
        onClose={handleCloseDetailsPanel}
        selectedItem={detailsPanelContentType === 'item' ? detailsPanelContentData : null}
        showFinalReport={detailsPanelContentType === 'report'}
        reportContent={detailsPanelContentType === 'report' ? detailsPanelContentData : null}
        query={currentQuery}
        isResearching={isResearching}
      />
    </div>
  );
}

export default App; 