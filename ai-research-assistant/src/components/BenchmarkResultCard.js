import React, { useState } from 'react';

const BenchmarkResultCard = ({ 
  question, 
  answer, 
  confidence, 
  confidenceLevel, 
  evidence, 
  sources, 
  expectedAnswer,
  isCorrect,
  reasoning,
  limitations 
}) => {
  const [showDetails, setShowDetails] = useState(false);
  
  // Get confidence color and icon
  const getConfidenceDisplay = (level, numeric) => {
    switch(level?.toUpperCase()) {
      case 'HIGH':
        return {
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          ),
          percentage: Math.round((numeric || 0.9) * 100)
        };
      case 'MEDIUM':
        return {
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ),
          percentage: Math.round((numeric || 0.6) * 100)
        };
      case 'LOW':
        return {
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          ),
          percentage: Math.round((numeric || 0.3) * 100)
        };
      default:
        return {
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
          ),
          percentage: Math.round((numeric || 0.5) * 100)
        };
    }
  };

  const confidenceDisplay = getConfidenceDisplay(confidenceLevel, confidence);

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Research Question</h3>
              <p className="text-sm text-gray-600">Benchmark Mode</p>
            </div>
          </div>
          {isCorrect !== undefined && (
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
              isCorrect 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              {isCorrect ? (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              )}
              {isCorrect ? 'Correct' : 'Incorrect'}
            </div>
          )}
        </div>
      </div>

      {/* Question */}
      <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
        <p className="text-gray-800 font-medium leading-relaxed">{question}</p>
      </div>

      {/* Answer Section */}
      <div className="px-6 py-6">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <div className="p-2 bg-green-100 rounded-lg">
              <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div className="flex-1">
            <h4 className="text-lg font-semibold text-gray-900 mb-2">Answer</h4>
            <p className="text-gray-800 leading-relaxed text-base">{answer}</p>
          </div>
        </div>

        {/* Confidence Section */}
        <div className="mt-6 flex items-center justify-between">
          {/* Toggle Details Button */}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center space-x-2 px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
          >
            <span className="font-medium">
              {showDetails ? 'Hide Details' : 'Show Details'}
            </span>
            <svg 
              className={`w-4 h-4 transform transition-transform ${showDetails ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* Expandable Details */}
        {showDetails && (
          <div className="mt-6 space-y-6 border-t border-gray-200 pt-6">
            {/* Evidence */}
            {evidence && (
              <div>
                <h5 className="font-semibold text-gray-900 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Supporting Evidence
                </h5>
                <p className="text-gray-700 bg-blue-50 p-4 rounded-lg border border-blue-200">{evidence}</p>
              </div>
            )}

            {/* Reasoning */}
            {reasoning && (
              <div>
                <h5 className="font-semibold text-gray-900 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  Reasoning
                </h5>
                <p className="text-gray-700 bg-purple-50 p-4 rounded-lg border border-purple-200">{reasoning}</p>
              </div>
            )}

            {/* Sources */}
            {sources && sources.length > 0 && (
              <div>
                <h5 className="font-semibold text-gray-900 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  Sources ({sources.length})
                </h5>
                <div className="space-y-2">
                  {sources.map((source, index) => (
                    <div key={index} className="flex items-center space-x-2 text-sm">
                      <span className="flex-shrink-0 w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center font-medium">
                        {index + 1}
                      </span>
                      <span className="text-gray-700">{source}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Expected Answer (if available) */}
            {expectedAnswer && (
              <div>
                <h5 className="font-semibold text-gray-900 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Expected Answer
                </h5>
                <p className="text-gray-700 bg-orange-50 p-4 rounded-lg border border-orange-200">{expectedAnswer}</p>
              </div>
            )}

            {/* Limitations */}
            {limitations && limitations !== 'None' && (
              <div>
                <h5 className="font-semibold text-gray-900 mb-2 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                  Limitations
                </h5>
                <p className="text-gray-700 bg-yellow-50 p-4 rounded-lg border border-yellow-200">{limitations}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default BenchmarkResultCard; 