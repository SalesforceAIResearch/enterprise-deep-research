"use client"

import { useState } from "react"
import DatabaseUpload from "./DatabaseUpload"

// Model options for the dropdown
const MODEL_OPTIONS = [
  {
    key: "openai-o4-mini",
    label: "o4-mini",
    model: "o4-mini",
    icon: (
      <svg width="18" height="18" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58 28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z"
          fill="#10A37F"
        />
      </svg>
    ),
  },
  {
    key: "openai-o4-mini-high",
    label: "o4-mini-high",
    model: "o4-mini-high",
    icon: (
      <svg width="18" height="18" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58 28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z"
          fill="#FFD700"
        />
      </svg>
    ),
  },
  {
    key: "openai-o3-mini",
    label: "o3-mini",
    model: "o3-mini",
    icon: (
      <svg width="18" height="18" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58 28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z"
          fill="#10A37F"
        />
      </svg>
    ),
  },
  {
    key: "google-gemini",
    label: "gemini-2.5-pro",
    model: "gemini-2.5-pro",
    icon: (
      <svg width="18" height="18" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M14.0001 0L17.5001 10.5L28.0001 14L17.5001 17.5L14.0001 28L10.5001 17.5L0.000061 14L10.5001 10.5L14.0001 0Z"
          fill="url(#paint0_linear_gemini)"
        />
        <defs>
          <linearGradient
            id="paint0_linear_gemini"
            x1="0.000061"
            y1="14"
            x2="28.0001"
            y2="14"
            gradientUnits="userSpaceOnUse"
          >
            <stop stopColor="#8E54E9" />
            <stop offset="1" stopColor="#4776E6" />
          </linearGradient>
        </defs>
      </svg>
    ),
  },
  {
    key: "anthropic-claude-4",
    label: "claude-sonnet-4",
    model: "claude-sonnet-4",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M13.827 3.52h3.603L24 20h-3.603l-6.57-16.48zm-7.258 0h3.767L16.906 20h-3.674l-1.343-3.461H5.017l-1.344 3.46H0L6.57 3.522zm4.132 9.959L8.453 7.687 6.205 13.48H10.7z"
          fill="#BD5CFF"
        />
      </svg>
    ),
  },
  {
    key: "anthropic-claude",
    label: "claude-sonnet-3.7",
    model: "claude-3-7-sonnet",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M13.827 3.52h3.603L24 20h-3.603l-6.57-16.48zm-7.258 0h3.767L16.906 20h-3.674l-1.343-3.461H5.017l-1.344 3.46H0L6.57 3.522zm4.132 9.959L8.453 7.687 6.205 13.48H10.7z"
          fill="#BD5CFF"
        />
      </svg>
    ),
  },
]

// Suggested questions for different modes
const INVESTIGATE_QUERIES = [
  "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give the city name.",
  "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan–May 2018 as listed on the NIH website?",
  "A paper about AI regulation first submitted to arXiv in June 2022 shows a figure with three axes, each axis labelled at both ends. Which of those label words is also used to describe a type of society in a Physics & Society article submitted on 11 Aug 2016?"
]

// Suggested questions
const RESEARCH_REPORT_QUERIES = [
  "What are the latest developments in renewable energy technologies?",
  "What are the key considerations for enterprise AI adoption and implementation?",
  "How is artificial intelligence transforming healthcare?",
]

export default function InitialScreen({ onBeginResearch }) {
  const [question, setQuestion] = useState("")
  const [effortLevel, setEffortLevel] = useState("standard")
  const [selectedModel, setSelectedModel] = useState("google-gemini")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastSubmitTime, setLastSubmitTime] = useState(0)
  const [mode, setMode] = useState("report")
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [uploadedFileContents, setUploadedFileContents] = useState([])
  const [isDragging, setIsDragging] = useState(false)

  // New state for Investigate mode file analysis
  const [investigateFiles, setInvestigateFiles] = useState([])
  const [investigateFileContents, setInvestigateFileContents] = useState([])
  const [isInvestigateDragging, setIsInvestigateDragging] = useState(false)
  const [fileAnalysisResults, setFileAnalysisResults] = useState({})


  const processFiles = (files) => {
    if (files.length > 0) {
      const newFiles = []
      const newContents = []
      let filesProcessed = 0

      files.forEach((file) => {
        const allowedExtensions = [
          ".txt",
          ".md",
          ".markdown",
          ".text",
          ".rtf",
          ".csv",
          ".json",
          ".xml",
          ".html",
          ".htm",
          ".log",
          ".yaml",
          ".yml",
          ".db",
          ".sqlite",
          ".sqlite3",
        ]
        const fileExtension = "." + file.name.split(".").pop().toLowerCase()

        if (!allowedExtensions.includes(fileExtension)) {
          console.warn(`File ${file.name} is not a supported format. Skipping.`)
          filesProcessed++
          if (filesProcessed === files.length) {
            setUploadedFiles([...uploadedFiles, ...newFiles])
            setUploadedFileContents([...uploadedFileContents, ...newContents])
          }
          return
        }

        newFiles.push(file)

        // Handle database files differently
        if (['.db', '.sqlite', '.sqlite3', '.csv', '.json'].includes(fileExtension)) {
          // For database files, we'll upload them to the database API
          uploadDatabaseFile(file, filesProcessed, files.length, newFiles, newContents)
        } else {
          // For text files, read the content
          const reader = new FileReader()

          reader.onload = (e) => {
            const content = e.target.result
            newContents.push({
              filename: file.name,
              content: content,
              size: file.size,
              type: 'text'
            })
            filesProcessed++

            if (filesProcessed === files.length) {
              setUploadedFiles([...uploadedFiles, ...newFiles])
              setUploadedFileContents([...uploadedFileContents, ...newContents])
            }
          }

          reader.onerror = (e) => {
            console.error(`Error reading file ${file.name}:`, e)
            filesProcessed++

            if (filesProcessed === files.length) {
              setUploadedFiles([...uploadedFiles, ...newFiles])
              setUploadedFileContents([...uploadedFileContents, ...newContents])
            }
          }

          reader.readAsText(file)
        }
      })
    }
  }

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files)
    processFiles(files)
  }

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    processFiles(files)
  }

  const removeFile = (indexToRemove) => {
    setUploadedFiles(uploadedFiles.filter((_, index) => index !== indexToRemove))
    setUploadedFileContents(uploadedFileContents.filter((_, index) => index !== indexToRemove))
  }

  // New functions for Investigate mode file handling
  const processInvestigateFiles = async (files) => {
    if (files.length > 0) {
      const newFiles = []
      const newAnalysisResults = { ...fileAnalysisResults }

      for (const file of files) {
        const allowedExtensions = [
          ".txt", ".md", ".markdown", ".pdf", ".docx", ".csv", ".xlsx", ".xls",
          ".json", ".xml", ".jpg", ".jpeg", ".png", ".gif", ".mp3", ".wav", ".mp4"
        ]
        const fileExtension = "." + file.name.split(".").pop().toLowerCase()

        if (!allowedExtensions.includes(fileExtension)) {
          console.warn(`File ${file.name} is not a supported format for analysis. Skipping.`)
          continue
        }

        newFiles.push(file)

        // Set initial processing status
        newAnalysisResults[file.name] = {
          fileId: null,
          status: 'processing',
          content: null
        }

        // Try to upload and analyze file
        try {
          // Detect the correct API base URL
          // If we're on a development port (3000, 3001, etc.), use localhost:8000 for the backend
          const currentPort = window.location.port
          const isDevelopment = currentPort && (currentPort.startsWith('30') || currentPort === '3000' || currentPort === '3001')
          const apiBaseUrl = isDevelopment ? 'http://localhost:8000' : ''

          const formData = new FormData()
          formData.append('file', file)
          formData.append('analyze_immediately', 'true')
          formData.append('analysis_type', 'quick')


          const response = await fetch(`${apiBaseUrl}/api/files/upload`, {
            method: 'POST',
            body: formData
          })

          if (response.ok) {
            const result = await response.json()
            newAnalysisResults[file.name] = {
              fileId: result.file_id,
              status: 'processing',
              content: null
            }

            // Poll for analysis results
            pollAnalysisResult(result.file_id, file.name, apiBaseUrl)
          } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`)
          }
        } catch (error) {
          console.error(`Error uploading file ${file.name}:`, error)

          // Fallback: Basic file content reading for text files
          if (['.txt', '.md', '.markdown', '.json', '.csv'].includes(fileExtension)) {
            try {
              const content = await readFileContent(file)
              newAnalysisResults[file.name] = {
                status: 'completed',
                content: `File content preview: ${content.substring(0, 500)}${content.length > 500 ? '...' : ''}`,
                metadata: {
                  file_type: fileExtension.substring(1),
                  file_size: file.size,
                  fallback: true
                }
              }
            } catch (readError) {
              newAnalysisResults[file.name] = {
                status: 'error',
                content: `File Analysis API not available. Error: ${error.message}. Please ensure the backend server is running with file analysis endpoints.`
              }
            }
          } else {
            newAnalysisResults[file.name] = {
              status: 'error',
              content: `File Analysis API not available. Error: ${error.message}. Please ensure the backend server is running with file analysis endpoints.`
            }
          }
        }
      }

      setInvestigateFiles([...investigateFiles, ...newFiles])
      setFileAnalysisResults(newAnalysisResults)
    }
  }

  // Helper function to read text file content
  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target.result)
      reader.onerror = (e) => reject(e)
      reader.readAsText(file)
    })
  }

  const pollAnalysisResult = async (fileId, fileName, apiBaseUrl) => {
    const maxAttempts = 30 // 30 attempts * 2 seconds = 1 minute max
    let attempts = 0

    const poll = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/files/${fileId}/analysis`)

        if (response.ok) {
          const analysis = await response.json()

          setFileAnalysisResults(prev => ({
            ...prev,
            [fileName]: {
              ...prev[fileName],
              status: 'completed',
              content: analysis.content_description || analysis.analysis || analysis.description || 'Analysis completed but no content description available',
              metadata: analysis.metadata || {},
              fullAnalysis: analysis // Store the full response for debugging
            }
          }))
        } else if (response.status === 404 && attempts < maxAttempts) {
          attempts++
          setTimeout(poll, 2000)
        } else {
          console.error(`Failed to get analysis for ${fileName}:`, response.status, response.statusText)
          setFileAnalysisResults(prev => ({
            ...prev,
            [fileName]: {
              ...prev[fileName],
              status: 'error',
              content: `Analysis failed: HTTP ${response.status}`
            }
          }))
        }
      } catch (error) {
        console.error(`Error polling analysis for ${fileName}:`, error)
        setFileAnalysisResults(prev => ({
          ...prev,
          [fileName]: {
            ...prev[fileName],
            status: 'error',
            content: `Analysis error: ${error.message}`
          }
        }))
      }
    }

    poll()
  }

  const handleInvestigateFileChange = (event) => {
    const files = Array.from(event.target.files)
    processInvestigateFiles(files)
  }

  const handleInvestigateDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsInvestigateDragging(true)
  }

  const handleInvestigateDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsInvestigateDragging(false)
    }
  }

  const handleInvestigateDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleInvestigateDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsInvestigateDragging(false)

    const files = Array.from(e.dataTransfer.files)
    processInvestigateFiles(files)
  }

  const removeInvestigateFile = (indexToRemove) => {
    const fileToRemove = investigateFiles[indexToRemove]
    setInvestigateFiles(investigateFiles.filter((_, index) => index !== indexToRemove))

    // Remove from analysis results
    const newResults = { ...fileAnalysisResults }
    delete newResults[fileToRemove.name]
    setFileAnalysisResults(newResults)
  }

  const handleSubmit = () => {
    const now = Date.now()
    const DEBOUNCE_TIME = 3000

    if (isSubmitting) {
      return
    }

    if (now - lastSubmitTime < DEBOUNCE_TIME) {
      return
    }

    setIsSubmitting(true)
    setLastSubmitTime(now)

    const minimumEffort = effortLevel === "quick"
    const extraEffort = effortLevel === "high"

    const selectedModelOption = MODEL_OPTIONS.find((opt) => opt.key === selectedModel)

    const [provider] = selectedModelOption.key.split("-", 2)
    const modelName = selectedModelOption.model

    // Prepare file content for the research request
    const fileContent = mode === "ask" ?
      Object.entries(fileAnalysisResults)
        .filter(([_, result]) => result.status === 'completed' && result.content)
        .map(([fileName, result]) => ({
          filename: fileName,
          content: result.content,
          metadata: result.metadata
        })) :
      uploadedFileContents

    // Extract database information for the agent
    const databaseInfo = uploadedFileContents
      .filter(content => content.type === 'database' && !content.error)
      .map(content => ({
        filename: content.filename,
        database_id: content.database_id,
        tables: content.tables,
        type: 'database'
      }))

    onBeginResearch(
      question || "Please provide information on this topic",
      extraEffort,
      minimumEffort,
      mode === "benchmark", // Convert string mode to boolean for benchmark mode
      {
        provider: provider,
        model: modelName,
      },
      fileContent,
      databaseInfo, // Pass database information to the research agent
    )

    setTimeout(() => {
      setIsSubmitting(false)
    }, DEBOUNCE_TIME)
  }

  const selectSuggestion = (suggestion) => {
    setQuestion(suggestion)
  }


  const uploadDatabaseFile = async (file, filesProcessed, totalFiles, newFiles, newContents) => {
    try {
      const apiBaseUrl = getApiBaseUrl()
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${apiBaseUrl}/api/database/upload`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        newContents.push({
          filename: file.name,
          content: `Database uploaded successfully. Tables: ${result.tables.join(', ')}`,
          size: file.size,
          type: 'database',
          database_id: result.database_id,
          tables: result.tables
        })

        // Database uploaded successfully
      } else {
        const errorData = await response.json()
        newContents.push({
          filename: file.name,
          content: `Database upload failed: ${errorData.detail || response.statusText}`,
          size: file.size,
          type: 'database',
          error: true
        })
      }
    } catch (error) {
      newContents.push({
        filename: file.name,
        content: `Database upload error: ${error.message}`,
        size: file.size,
        type: 'database',
        error: true
      })
    }

    filesProcessed++
    if (filesProcessed === totalFiles) {
      setUploadedFiles([...uploadedFiles, ...newFiles])
      setUploadedFileContents([...uploadedFileContents, ...newContents])
    }
  }

  const getApiBaseUrl = () => {
    const currentPort = window.location.port
    const isDevelopment = currentPort && (currentPort.startsWith('30') || currentPort === '3000' || currentPort === '3001')
    return isDevelopment ? 'http://localhost:8000' : ''
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-blue-50 via-white to-[#f3f3f3] py-8 px-4">
      {/* Research Card */}
      <div className="max-w-3xl mx-auto bg-white/95 backdrop-blur-sm rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.12)] hover:shadow-[0_8px_40px_rgb(0,0,0,0.16)] border border-slate-200/60 overflow-hidden transition-all duration-300">
        {/* Card Header */}
        <div className="bg-gradient-to-r from-[#f3f3f3]/80 via-white/50 to-blue-50/50 border-b border-slate-200/60 px-8 py-5 backdrop-blur-sm">
          <div className="mb-2">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-[#032d60] to-[#0176d3] bg-clip-text text-transparent mb-0.5">
              Create Research Report
            </h2>
            <p className="text-sm text-slate-600">
              Generate comprehensive research reports with AI
            </p>
          </div>
        </div>

        {/* Card Body */}
        <div className="px-8 py-4 space-y-4 max-h-[calc(100vh-380px)] overflow-y-auto">


          {/* Question input */}
          <div className="space-y-2">
            <label
              htmlFor="question"
              className="flex items-center text-sm font-semibold text-[#032d60]"
            >
              <div className="w-8 h-8 rounded-lg bg-[#0176d3] flex items-center justify-center mr-2.5 shadow-sm">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              Your Research Question
            </label>
            <div className="relative group">
              <textarea
                id="question"
                rows="2"
                className="w-full px-4 py-3 bg-white/50 border-2 border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#0176d3]/50 focus:border-[#0176d3] resize-none text-slate-900 placeholder-slate-400 text-sm leading-relaxed transition-all duration-200 hover:border-[#0176d3]/30 hover:bg-white"
                placeholder="What is the current state of quantum computing and its potential applications?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
              <div className="absolute bottom-2 right-2 text-xs text-slate-400 font-medium bg-white/80 px-2 py-0.5 rounded-md">
                {question.length}/500
              </div>
            </div>
          </div>

          {/* File upload with enhanced design - only show in Report mode */}
          {mode === "report" && (
            <div className="space-y-3 flex-shrink-0">
              <label className="flex items-center text-sm font-semibold text-[#032d60]">
                <div className="w-8 h-8 rounded-lg bg-[#04844b] flex items-center justify-center mr-2.5 shadow-sm">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                Enterprise Data Upload (Optional)
              </label>
              <div
                className={`relative flex justify-center px-4 py-3 border-2 border-dashed rounded-xl transition-all duration-300 ${isDragging
                  ? "border-[#04844b] bg-gradient-to-br from-green-50 to-[#e6f5ef] scale-[1.02] shadow-xl shadow-[#04844b]/20"
                  : "border-slate-300 hover:border-[#04844b]/50 bg-gradient-to-br from-white/80 to-slate-50/80 hover:shadow-lg backdrop-blur-sm"
                  }`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <div className="space-y-1.5 text-center">
                  <div
                    className={`mx-auto w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${isDragging
                      ? "bg-[#04844b] shadow-lg scale-110"
                      : "bg-gradient-to-br from-slate-100 to-slate-200"
                      }`}
                  >
                    <svg
                      className={`w-5 h-5 transition-colors ${isDragging ? "text-white" : "text-slate-500"}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <div className="space-y-0.5">
                    <div className="flex text-xs text-slate-700 justify-center items-center gap-1">
                      <label
                        htmlFor="file-upload-input"
                        className="relative cursor-pointer bg-[#e6f5ef] hover:bg-[#c2e6d5] rounded-md px-2.5 py-0.5 font-semibold text-[#04844b] hover:text-[#025c30] focus-within:outline-none focus-within:ring-2 focus-within:ring-[#04844b] transition-all"
                      >
                        <span>Upload files</span>
                        <input
                          id="file-upload-input"
                          name="file-upload-input"
                          type="file"
                          className="sr-only"
                          accept=".txt,.md,.markdown,.text,.rtf,.csv,.json,.xml,.html,.htm,.log,.yaml,.yml,.db,.sqlite,.sqlite3"
                          multiple
                          onChange={handleFileChange}
                        />
                      </label>
                      <span className="text-xs">or drag and drop</span>
                    </div>
                    <p className={`text-xs transition-colors ${isDragging ? "text-[#04844b]" : "text-slate-500"}`}>
                      {isDragging ? "Drop files here!" : "Text files, databases (.db, .sqlite, .csv, .json), up to 10MB each"}
                    </p>
                  </div>

                  {uploadedFiles.length > 0 && (
                    <div className="mt-6 space-y-2">
                      {uploadedFiles.map((file, index) => {
                        const fileContent = uploadedFileContents[index]
                        const isDatabase = fileContent && fileContent.type === 'database'
                        const fileExtension = '.' + file.name.split('.').pop().toLowerCase()
                        const isDbFile = ['.db', '.sqlite', '.sqlite3', '.csv', '.json'].includes(fileExtension)

                        return (
                          <div
                            key={index}
                            className={`flex items-center justify-between text-xs rounded-lg px-4 py-3 shadow-sm border ${isDatabase ? 'bg-blue-50 border-blue-200' : 'bg-white border-slate-200'
                              }`}
                          >
                            <div className="flex items-center space-x-3">
                              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isDatabase ? 'bg-blue-100' : 'bg-slate-100'
                                }`}>
                                {isDatabase ? (
                                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                                  </svg>
                                ) : (
                                  <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                  </svg>
                                )}
                              </div>
                              <div>
                                <span className={`font-medium ${isDatabase ? 'text-blue-700' : 'text-slate-700'}`}>
                                  {file.name}
                                </span>
                                {isDatabase && fileContent && !fileContent.error && (
                                  <div className="text-xs text-blue-600 mt-1">
                                    Database • {fileContent.tables ? fileContent.tables.length : 0} tables
                                  </div>
                                )}
                                {isDatabase && fileContent && fileContent.error && (
                                  <div className="text-xs text-red-600 mt-1">
                                    Upload failed
                                  </div>
                                )}
                              </div>
                            </div>
                            <button
                              onClick={() => removeFile(index)}
                              className="w-6 h-6 text-slate-400 hover:text-red-500 transition-colors rounded-full hover:bg-red-50 flex items-center justify-center"
                              aria-label={`Remove ${file.name}`}
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Suggested questions */}
          <div className="space-y-3">
            <label className="flex items-center text-sm font-semibold text-[#032d60]">
              <div className="w-8 h-8 rounded-lg bg-[#ea7600] flex items-center justify-center mr-2.5 shadow-sm">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              Example Questions
            </label>
            <div className="space-y-2">
              {mode === "ask" && INVESTIGATE_QUERIES.map((suggestion, index) => (
                <button
                  key={index}
                  className="group w-full text-left p-3 bg-gradient-to-br from-white to-blue-50/30 border border-slate-200 rounded-xl hover:border-[#0176d3] hover:shadow-md hover:scale-[1.02] transition-all duration-200 text-sm text-slate-700 flex items-start gap-3"
                  onClick={() => selectSuggestion(suggestion)}
                  aria-label={`Select suggestion: ${suggestion}`}
                >
                  <div className="w-6 h-6 rounded-lg bg-[#0176d3] flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform shadow-sm">
                    <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <span className="flex-1 leading-relaxed group-hover:text-[#032d60] transition-colors">{suggestion}</span>
                </button>
              ))}
              {mode === "report" && RESEARCH_REPORT_QUERIES.map((suggestion, index) => (
                <button
                  key={index}
                  className="group w-full text-left p-3 bg-gradient-to-br from-white to-green-50/30 border border-slate-200 rounded-xl hover:border-[#04844b] hover:shadow-md hover:scale-[1.02] transition-all duration-200 text-sm text-slate-700 flex items-start gap-3"
                  onClick={() => selectSuggestion(suggestion)}
                  aria-label={`Select suggestion: ${suggestion}`}
                >
                  <div className="w-6 h-6 rounded-lg bg-[#04844b] flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform shadow-sm">
                    <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <span className="flex-1 leading-relaxed group-hover:text-[#032d60] transition-colors">{suggestion}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Control Panel - Attached to card */}
        <div className="flex-shrink-0 border-t border-slate-200/60 bg-gradient-to-r from-[#f3f3f3]/80 via-white/50 to-slate-50/50 px-8 py-5 backdrop-blur-sm">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              {/* Model selector */}
              <div className="flex items-center bg-white/90 backdrop-blur-sm border border-slate-200/60 rounded-xl overflow-hidden shadow-md hover:shadow-lg transition-all duration-200">
                <div className="flex items-center justify-center w-11 h-11 bg-gradient-to-br from-slate-50 to-slate-100 border-r border-slate-200">
                  {MODEL_OPTIONS.find((opt) => opt.key === selectedModel)?.icon}
                </div>
                <div className="relative">
                  <select
                    id="model-switcher"
                    className="appearance-none bg-transparent border-none pl-3 pr-8 py-2.5 text-slate-800 font-semibold focus:outline-none text-sm cursor-pointer"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    aria-label="Select model"
                  >
                    {MODEL_OPTIONS.map((option) => (
                      <option key={option.key} value={option.key}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-slate-400">
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </div>

              {/* Effort level */}
              {mode === "report" && (
                <div className="flex items-center bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl p-1.5 gap-1 shadow-md">
                  <button
                    onClick={() => setEffortLevel("quick")}
                    className={`px-5 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${effortLevel === "quick"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-300 hover:text-white hover:bg-slate-700/50"
                      }`}
                    aria-pressed={effortLevel === "quick"}
                    title="Fast results with fewer sources - ideal for quick insights"
                  >
                    Quick
                  </button>
                  <button
                    onClick={() => setEffortLevel("standard")}
                    className={`px-5 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${effortLevel === "standard"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-300 hover:text-white hover:bg-slate-700/50"
                      }`}
                    aria-pressed={effortLevel === "standard"}
                    title="Balanced approach with moderate depth and quality sources"
                  >
                    Standard
                  </button>
                  <button
                    onClick={() => setEffortLevel("high")}
                    className={`px-5 py-2 text-sm font-semibold rounded-lg transition-all duration-200 ${effortLevel === "high"
                      ? "bg-white text-slate-900 shadow-sm"
                      : "text-slate-300 hover:text-white hover:bg-slate-700/50"
                      }`}
                    aria-pressed={effortLevel === "high"}
                    title="Comprehensive research with extensive sources and thorough analysis"
                  >
                    Deep
                  </button>
                </div>
              )}
            </div>

            {/* Submit button */}
            <button
              className={`flex items-center justify-center gap-2 px-8 py-3 rounded-xl font-bold text-sm transition-all duration-200 ${isSubmitting
                ? "bg-slate-400 cursor-not-allowed text-white shadow-lg"
                : "bg-[#0176d3] hover:bg-[#014486] text-white shadow-lg shadow-[#0176d3]/30 hover:shadow-xl hover:shadow-[#0176d3]/40 hover:scale-105 active:scale-95"
                }`}
              onClick={handleSubmit}
              disabled={isSubmitting}
              aria-label="Start Research"
              aria-disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Start Research</span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="group-hover:translate-x-1 transition-transform">
                    <path d="M13 5l7 7-7 7M5 5l7 7-7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

