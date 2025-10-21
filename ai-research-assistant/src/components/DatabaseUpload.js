import React, { useState, useCallback } from 'react';

const DatabaseUpload = ({ onDatabaseUploaded, onQueryExecuted }) => {
  const [databases, setDatabases] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryResults, setQueryResults] = useState(null);
  const [error, setError] = useState('');

  // Detect the correct API base URL
  const getApiBaseUrl = () => {
    const currentPort = window.location.port;
    const isDevelopment = currentPort && (currentPort.startsWith('30') || currentPort === '3000' || currentPort === '3001');
    return isDevelopment ? 'http://localhost:8000' : '';
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  };

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    handleFileUpload(files);
  };

  const handleFileUpload = async (files) => {
    if (files.length === 0) return;

    const file = files[0]; // Only handle one file at a time
    const allowedExtensions = ['.db', '.sqlite', '.sqlite3', '.csv'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(fileExtension)) {
      setError(`Unsupported file type: ${fileExtension}. Allowed types: ${allowedExtensions.join(', ')}`);
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/database/upload`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        
        // Add to databases list
        const newDatabase = {
          id: result.database_id,
          filename: result.filename,
          file_type: result.file_type,
          tables: result.tables,
          uploaded_at: new Date().toISOString()
        };

        setDatabases(prev => [...prev, newDatabase]);
        setSelectedDatabase(result.database_id);
        
        if (onDatabaseUploaded) {
          onDatabaseUploaded(newDatabase);
        }
      } else {
        const errorData = await response.json();
        setError(`Upload failed: ${errorData.detail || response.statusText}`);
      }
    } catch (error) {
      setError(`Upload error: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim() || !selectedDatabase) {
      setError('Please enter a query and select a database');
      return;
    }

    setIsQuerying(true);
    setError('');
    setQueryResults(null);

    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/database/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          database_id: selectedDatabase
        })
      });

      if (response.ok) {
        const result = await response.json();
        setQueryResults(result);
        
        if (onQueryExecuted) {
          onQueryExecuted(result);
        }
      } else {
        const errorData = await response.json();
        setError(`Query failed: ${errorData.detail || response.statusText}`);
      }
    } catch (error) {
      setError(`Query error: ${error.message}`);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleDeleteDatabase = async (databaseId) => {
    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/database/${databaseId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setDatabases(prev => prev.filter(db => db.id !== databaseId));
        if (selectedDatabase === databaseId) {
          setSelectedDatabase('');
        }
        setQueryResults(null);
      } else {
        const errorData = await response.json();
        setError(`Delete failed: ${errorData.detail || response.statusText}`);
      }
    } catch (error) {
      setError(`Delete error: ${error.message}`);
    }
  };

  const formatResults = (results) => {
    if (!results || results.error) {
      return <div className="text-red-600">Error: {results?.error || 'Unknown error'}</div>;
    }

    if (results.results?.type === 'select') {
      const { columns, rows } = results.results;
      
      return (
        <div className="space-y-4">
          <div className="bg-gray-50 p-3 rounded-lg">
            <h4 className="font-semibold text-sm text-gray-700 mb-2">Generated SQL:</h4>
            <code className="text-sm bg-white p-2 rounded border block overflow-x-auto">
              {results.sql}
            </code>
          </div>
          
          <div className="bg-white border rounded-lg overflow-hidden">
            <div className="bg-gray-50 px-4 py-2 border-b">
              <h4 className="font-semibold text-sm text-gray-700">
                Results ({rows.length} rows)
              </h4>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {columns.map((column, index) => (
                      <th
                        key={index}
                        className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                      >
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {rows.slice(0, 10).map((row, rowIndex) => (
                    <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {columns.map((column, colIndex) => (
                        <td key={colIndex} className="px-4 py-2 text-sm text-gray-900">
                          {row[column] !== null && row[column] !== undefined ? String(row[column]) : 'NULL'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length > 10 && (
                <div className="px-4 py-2 bg-gray-50 text-sm text-gray-500">
                  Showing first 10 of {rows.length} rows
                </div>
              )}
            </div>
          </div>
        </div>
      );
    } else if (results.results?.type === 'modify') {
      return (
        <div className="space-y-4">
          <div className="bg-gray-50 p-3 rounded-lg">
            <h4 className="font-semibold text-sm text-gray-700 mb-2">Generated SQL:</h4>
            <code className="text-sm bg-white p-2 rounded border block overflow-x-auto">
              {results.sql}
            </code>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-green-800 font-medium">
                Query executed successfully. {results.results.rows_affected} rows affected.
              </span>
            </div>
          </div>
        </div>
      );
    }

    return <div className="text-gray-600">No results to display</div>;
  };

  return (
    <div className="space-y-6">
      {/* Database Upload Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
          </svg>
          Upload Database
        </h3>
        
        <div
          className={`relative flex justify-center px-6 py-8 border-2 border-dashed rounded-lg transition-all duration-300 ${
            isDragging
              ? 'border-blue-500 bg-blue-50 scale-105'
              : 'border-gray-300 hover:border-blue-400 bg-gray-50'
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <div className="text-center">
            <div className={`mx-auto w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${
              isDragging ? 'bg-blue-500' : 'bg-gray-200'
            }`}>
              <svg className={`w-6 h-6 ${isDragging ? 'text-white' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-center gap-2">
                <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors">
                  {isUploading ? 'Uploading...' : 'Choose file'}
                  <input
                    type="file"
                    className="sr-only"
                    accept=".db,.sqlite,.sqlite3,.csv"
                    onChange={handleFileChange}
                    disabled={isUploading}
                  />
                </label>
                <span className="text-sm text-gray-500">or drag and drop</span>
              </div>
              <p className={`text-sm transition-colors ${isDragging ? 'text-blue-600' : 'text-gray-500'}`}>
                {isDragging ? 'Drop your database file here!' : 'SQLite (.db, .sqlite, .sqlite3) or CSV files'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Database List */}
      {databases.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Uploaded Databases
          </h3>
          
          <div className="space-y-3">
            {databases.map((db) => (
              <div key={db.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                    </svg>
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">{db.filename}</div>
                    <div className="text-sm text-gray-500">
                      {db.file_type.toUpperCase()} • {db.tables.length} table{db.tables.length !== 1 ? 's' : ''}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleDeleteDatabase(db.id)}
                    className="text-red-600 hover:text-red-800 p-1 rounded"
                    title="Delete database"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Query Section */}
      {databases.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            Text2SQL Query
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Database
              </label>
              <select
                value={selectedDatabase}
                onChange={(e) => setSelectedDatabase(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Choose a database...</option>
                {databases.map((db) => (
                  <option key={db.id} value={db.id}>
                    {db.filename} ({db.file_type.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Natural Language Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., Show me all customers from California with orders over $1000"
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <button
              onClick={handleQuery}
              disabled={!query.trim() || !selectedDatabase || isQuerying}
              className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors flex items-center justify-center"
            >
              {isQuerying ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Executing Query...
                </>
              ) : (
                'Execute Query'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Query Results */}
      {queryResults && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Query Results
          </h3>
          {formatResults(queryResults)}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-red-800 font-medium">Error</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
          <button
            onClick={() => setError('')}
            className="mt-2 text-red-600 hover:text-red-800 text-sm underline"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
};

export default DatabaseUpload;
