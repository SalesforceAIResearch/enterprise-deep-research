"use client"

import { useState } from "react"

// Model options for the dropdown
const MODEL_OPTIONS = [
  {
    key: "openai-o4-mini",
    label: "o4-mini",
    model: "o4-mini",
    icon: (
      <svg width="20" height="20" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg width="20" height="20" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg width="20" height="20" viewBox="0 0 256 260" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg width="20" height="20" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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

const RESEARCH_REPORT_QUERIES = [
  "Salesforce investment thesis and its AI strategy",
  "Who is Silvio Savarese?",
  "What is LangGraph and show me a few examples?",
]

export default function InitialScreen({ onBeginResearch }) {
  const [question, setQuestion] = useState("")
  const [effortLevel, setEffortLevel] = useState("standard")
  const [selectedModel, setSelectedModel] = useState("anthropic-claude-4")
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
        ]
        const fileExtension = "." + file.name.split(".").pop().toLowerCase()

        if (!allowedExtensions.includes(fileExtension)) {
          console.warn(`File ${file.name} is not a supported text format. Skipping.`)
          filesProcessed++
          if (filesProcessed === files.length) {
            setUploadedFiles([...uploadedFiles, ...newFiles])
            setUploadedFileContents([...uploadedFileContents, ...newContents])
          }
          return
        }

        newFiles.push(file)
        const reader = new FileReader()

        reader.onload = (e) => {
          const content = e.target.result
          newContents.push({
            filename: file.name,
            content: content,
            size: file.size,
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

          console.log(`Attempting to upload ${file.name} to ${apiBaseUrl}/api/files/upload`)
          console.log(`Current port: ${currentPort}, isDevelopment: ${isDevelopment}`)

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
        console.log(`Polling analysis for ${fileName} (attempt ${attempts + 1})`)
        const response = await fetch(`${apiBaseUrl}/api/files/${fileId}/analysis`)
        
        if (response.ok) {
          const analysis = await response.json()
          console.log(`Analysis response for ${fileName}:`, analysis)
          
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
          console.log(`Analysis not ready for ${fileName}, retrying in 2 seconds...`)
          setTimeout(poll, 2000) // Poll every 2 seconds
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
      console.log("InitialScreen: Already submitting, ignoring click")
      return
    }

    if (now - lastSubmitTime < DEBOUNCE_TIME) {
      console.log("InitialScreen: Too soon since last submit, ignoring")
      return
    }

    setIsSubmitting(true)
    setLastSubmitTime(now)

    console.log("InitialScreen: handleSubmit called!")

    const minimumEffort = effortLevel === "quick"
    const extraEffort = effortLevel === "high"

    const selectedModelOption = MODEL_OPTIONS.find((opt) => opt.key === selectedModel)
    console.log("InitialScreen: selected model option:", selectedModelOption)

    const [provider] = selectedModelOption.key.split("-", 2)
    const modelName = selectedModelOption.model
    console.log("InitialScreen: provider:", provider, "model:", modelName)

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

    console.log("InitialScreen: calling onBeginResearch with:", {
      question: question || "Please provide information on this topic",
      extraEffort,
      minimumEffort,
      mode,
      modelConfig: {
        provider,
        model: modelName,
      },
      uploadedFileContents: fileContent,
    })

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
    )

    setTimeout(() => {
      setIsSubmitting(false)
    }, DEBOUNCE_TIME)
  }

  const selectSuggestion = (suggestion) => {
    setQuestion(suggestion)
  }

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-slate-50 via-white to-slate-50/50 rounded-xl border border-slate-200/60 shadow-xl shadow-slate-900/5 overflow-hidden backdrop-blur-sm">
      {/* Header with gradient and improved typography */}
      <div className="px-4 py-1 sm:px-6 sm:py-1 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700/50">
        <div className="flex items-center space-x-3">
          <svg className="w-8 h-8 text-blue-400" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M18.9 11.1c-.1-2.4-2.1-4.4-4.6-4.4-1.4 0-2.7.6-3.6 1.7-.4-.2-.9-.3-1.4-.3-1.8 0-3.3 1.5-3.3 3.3 0 .3 0 .5.1.8C4.8 12.6 4 13.7 4 15c0 1.7 1.3 3 3 3h11c1.7 0 3-1.3 3-3 0-1.4-.9-2.6-2.1-2.9z"/>
          </svg>
          <div style={{ marginTop: '-10px' }}>
            <h1 className="text-lg sm:text-xl font-bold text-white tracking-tight">Salesforce AI Research</h1>
          </div>
        </div>
      </div>

      {/* Main content with improved spacing and modern cards */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 flex justify-center">
        <div className="max-w-2xl w-full space-y-4 sm:space-y-6">
          {/* Main research card with modern styling */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg shadow-slate-900/5 border border-slate-200/60 p-4 sm:p-6 hover:shadow-xl transition-all duration-300">
            <div className="space-y-4">
              {/* Header section */}
              <div className="space-y-3">
                <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">Ask a research question</h2>
              </div>

              {/* Question input with modern styling */}
              <div className="space-y-3">
                <label
                  htmlFor="question"
                  className="block text-xs font-semibold text-slate-700 tracking-wide uppercase"
                >
                  Research Question
                </label>
                <div className="relative">
                  <textarea
                    id="question"
                    rows="3"
                    className="w-full p-4 bg-slate-50/50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 resize-none transition-all duration-200 shadow-sm text-slate-900 placeholder-slate-400 text-base leading-relaxed"
                    placeholder="What is the current state of quantum computing and its potential impact on enterprise software?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                  />
                  <div className="absolute bottom-3 right-3 text-xs text-slate-400 font-medium">
                    {question.length}/500
                  </div>
                </div>
              </div>

              {/* File upload and analysis for Investigate mode */}
              {mode === "ask" && (
                <div className="space-y-3">
                  <label className="block text-xs font-semibold text-slate-700 tracking-wide uppercase">
                    Upload File for Analysis
                  </label>
                  
                  <div
                    className={`relative flex justify-center px-4 py-6 border-2 border-dashed rounded-xl transition-all duration-300 ${
                      isInvestigateDragging
                        ? "border-blue-400 bg-blue-50/50 scale-[1.02]"
                        : "border-slate-300 hover:border-slate-400 bg-slate-50/30"
                    }`}
                    onDragEnter={handleInvestigateDragEnter}
                    onDragLeave={handleInvestigateDragLeave}
                    onDragOver={handleInvestigateDragOver}
                    onDrop={handleInvestigateDrop}
                  >
                    <div className="space-y-4 text-center">
                      <div
                        className={`mx-auto w-10 h-10 rounded-full flex items-center justify-center ${
                          isInvestigateDragging ? "bg-blue-100" : "bg-slate-100"
                        }`}
                      >
                        <svg
                          className={`w-6 h-6 ${isInvestigateDragging ? "text-blue-600" : "text-slate-500"}`}
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
                      <div className="space-y-2">
                        <div className="flex text-sm text-slate-600 justify-center">
                          <label
                            htmlFor="investigate-file-upload-input"
                            className="relative cursor-pointer bg-white rounded-lg px-3 py-1 font-semibold text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 transition-colors"
                          >
                            <span>Upload file</span>
                            <input
                              id="investigate-file-upload-input"
                              name="investigate-file-upload-input"
                              type="file"
                              className="sr-only"
                              accept=".txt,.md,.markdown,.pdf,.docx,.csv,.xlsx,.xls,.json,.xml,.jpg,.jpeg,.png,.gif,.mp3,.wav,.mp4"
                              onChange={handleInvestigateFileChange}
                            />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className={`text-xs ${isInvestigateDragging ? "text-blue-600" : "text-slate-500"}`}>
                          {isInvestigateDragging ? "Drop file here!" : "PDF, Word, CSV, Images, Audio & more"}
                        </p>
                      </div>

                      {investigateFiles.length > 0 && (
                        <div className="mt-6 space-y-3">
                          {investigateFiles.map((file, index) => {
                            const analysisResult = fileAnalysisResults[file.name]
                            return (
                              <div
                                key={index}
                                className="text-left bg-white rounded-lg p-4 shadow-sm border border-slate-200"
                              >
                                <div className="flex items-center justify-between mb-3">
                                  <div className="flex items-center space-x-3">
                                    <div className="w-8 h-8 bg-slate-100 rounded-lg flex items-center justify-center">
                                      <svg
                                        className="w-4 h-4 text-slate-600"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          strokeWidth="2"
                                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                        />
                                      </svg>
                                    </div>
                                    <div>
                                      <span className="font-medium text-slate-700 text-sm">{file.name}</span>
                                      <div className="flex items-center space-x-2 mt-1">
                                        {analysisResult?.status === 'processing' && (
                                          <div className="flex items-center space-x-1">
                                            <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                                            <span className="text-xs text-blue-600">Analyzing...</span>
                                          </div>
                                        )}
                                        {analysisResult?.status === 'completed' && (
                                          <div className="flex items-center space-x-1">
                                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                            <span className="text-xs text-green-600">Analysis complete</span>
                                          </div>
                                        )}
                                        {analysisResult?.status === 'error' && (
                                          <div className="flex items-center space-x-1">
                                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                            <span className="text-xs text-red-600">Analysis failed</span>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <button
                                    onClick={() => removeInvestigateFile(index)}
                                    className="w-6 h-6 text-slate-400 hover:text-red-500 transition-colors rounded-full hover:bg-red-50 flex items-center justify-center"
                                    aria-label={`Remove ${file.name}`}
                                  >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="2"
                                        d="M6 18L18 6M6 6l12 12"
                                      />
                                    </svg>
                                  </button>
                                </div>
                                
                                {analysisResult?.content && (
                                  <div className="mt-3 p-3 bg-slate-50 rounded-lg">
                                    <h4 className="text-xs font-semibold text-slate-700 mb-2 uppercase tracking-wide flex items-center">
                                      <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                      </svg>
                                      {analysisResult.metadata?.fallback ? 'Content Preview' : 'AI Analysis Results'}
                                    </h4>
                                    <div className="text-sm text-slate-700 leading-relaxed">
                                      {analysisResult.content.length > 500 ? (
                                        <div>
                                          <p className="mb-2">{analysisResult.content}</p>
                                        </div>
                                      ) : (
                                        <p>{analysisResult.content}</p>
                                      )}
                                    </div>
                                    {analysisResult.metadata && Object.keys(analysisResult.metadata).length > 0 && (
                                      <div className="mt-3">
                                        <h5 className="text-xs font-medium text-slate-600 mb-2">File Details</h5>
                                        <div className="flex flex-wrap gap-2">
                                          {Object.entries(analysisResult.metadata).slice(0, 5).map(([key, value]) => {
                                            const displayValue = typeof value === 'object' ? 
                                              JSON.stringify(value).slice(0, 30) + (JSON.stringify(value).length > 30 ? '...' : '') : 
                                              String(value).slice(0, 30) + (String(value).length > 30 ? '...' : '')
                                            
                                            return (
                                              <span
                                                key={key}
                                                className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700"
                                                title={`${key}: ${value}`}
                                              >
                                                <span className="font-medium">{key}:</span>
                                                <span className="ml-1">{displayValue}</span>
                                              </span>
                                            )
                                          })}
                                        </div>
                                      </div>
                                    )}
                                    {analysisResult.metadata?.fallback && (
                                      <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-700">
                                        <strong>Note:</strong> This is a content preview. For full AI analysis, ensure the backend server is running.
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* File upload with enhanced design - only show in Report mode */}
              {mode === "report" && (
                <div className="space-y-3">
                  <label className="block text-xs font-semibold text-slate-700 tracking-wide uppercase">
                    External Knowledge
                  </label>
                  <div
                    className={`relative flex justify-center px-4 py-6 border-2 border-dashed rounded-xl transition-all duration-300 ${
                      isDragging
                        ? "border-blue-400 bg-blue-50/50 scale-[1.02]"
                        : "border-slate-300 hover:border-slate-400 bg-slate-50/30"
                    }`}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                  >
                    <div className="space-y-4 text-center">
                      <div
                        className={`mx-auto w-10 h-10 rounded-full flex items-center justify-center ${
                          isDragging ? "bg-blue-100" : "bg-slate-100"
                        }`}
                      >
                        <svg
                          className={`w-6 h-6 ${isDragging ? "text-blue-600" : "text-slate-500"}`}
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
                      <div className="space-y-2">
                        <div className="flex text-sm text-slate-600 justify-center">
                          <label
                            htmlFor="file-upload-input"
                            className="relative cursor-pointer bg-white rounded-lg px-3 py-1 font-semibold text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 transition-colors"
                          >
                            <span>Upload files</span>
                            <input
                              id="file-upload-input"
                              name="file-upload-input"
                              type="file"
                              className="sr-only"
                              accept=".txt,.md,.markdown,.text,.rtf,.csv,.json,.xml,.html,.htm,.log,.yaml,.yml"
                              multiple
                              onChange={handleFileChange}
                            />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className={`text-xs ${isDragging ? "text-blue-600" : "text-slate-500"}`}>
                          {isDragging ? "Drop files here!" : "Up to 10MB each"}
                        </p>
                      </div>

                      {uploadedFiles.length > 0 && (
                        <div className="mt-6 space-y-2">
                          {uploadedFiles.map((file, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between text-xs bg-white rounded-lg px-4 py-3 shadow-sm border border-slate-200"
                            >
                              <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 bg-slate-100 rounded-lg flex items-center justify-center">
                                  <svg
                                    className="w-4 h-4 text-slate-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth="2"
                                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                    />
                                  </svg>
                                </div>
                                <span className="font-medium text-slate-700">{file.name}</span>
                              </div>
                              <button
                                onClick={() => removeFile(index)}
                                className="w-6 h-6 text-slate-400 hover:text-red-500 transition-colors rounded-full hover:bg-red-50 flex items-center justify-center"
                                aria-label={`Remove ${file.name}`}
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M6 18L18 6M6 6l12 12"
                                  />
                                </svg>
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Suggested questions with modern card design */}
              <div className="space-y-4">
                <h3 className="text-xs font-semibold text-slate-700 tracking-wide uppercase">Suggested Questions</h3>
                <div className="grid gap-3">
                  {mode === "ask" && INVESTIGATE_QUERIES.map((suggestion, index) => (
                    <button
                      key={index}
                      className="group w-full text-left p-3 bg-gradient-to-r from-slate-50 to-slate-50/50 border border-slate-200 rounded-xl hover:from-blue-50 hover:to-blue-50/50 hover:border-blue-200 transition-all duration-200 shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
                      onClick={() => selectSuggestion(suggestion)}
                      aria-label={`Select suggestion: ${suggestion}`}
                    >
                      <div className="flex items-start space-x-3">
                        <div className="w-6 h-6 mt-0.5 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                          <svg
                            className="w-3.5 h-3.5 text-blue-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M13 10V3L4 14h7v7l9-11h-7z"
                            />
                          </svg>
                        </div>
                        <span className="text-slate-700 group-hover:text-slate-900 font-medium leading-relaxed text-xs sm:text-sm">
                          {suggestion}
                        </span>
                      </div>
                    </button>
                  ))}
                  {mode === "report" && RESEARCH_REPORT_QUERIES.map((suggestion, index) => (
                    <button
                      key={index}
                      className="group w-full text-left p-3 bg-gradient-to-r from-slate-50 to-slate-50/50 border border-slate-200 rounded-xl hover:from-blue-50 hover:to-blue-50/50 hover:border-blue-200 transition-all duration-200 shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
                      onClick={() => selectSuggestion(suggestion)}
                      aria-label={`Select suggestion: ${suggestion}`}
                    >
                      <div className="flex items-start space-x-3">
                        <div className="w-6 h-6 mt-0.5 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                          <svg
                            className="w-3.5 h-3.5 text-blue-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M13 10V3L4 14h7v7l9-11h-7z"
                            />
                          </svg>
                        </div>
                        <span className="text-slate-700 group-hover:text-slate-900 font-medium leading-relaxed text-xs sm:text-sm">
                          {suggestion}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced footer with modern controls */}
      <div className="border-t border-slate-200/60 bg-white/80 backdrop-blur-sm p-4">
        <div className="flex justify-between items-center max-w-3xl mx-auto gap-3 sm:gap-4">
          <div className="flex items-center gap-4 flex-wrap">
            {/* Model selector with enhanced design */}
            <div className="flex items-center h-10 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden hover:shadow-md transition-shadow">
              <div className="flex items-center justify-center w-10 h-10 bg-slate-50">
                {MODEL_OPTIONS.find((opt) => opt.key === selectedModel)?.icon}
              </div>
              <div className="relative pr-3">
                <select
                  id="model-switcher"
                  className="appearance-none bg-transparent border-none pr-8 py-2 pl-3 text-slate-700 font-semibold focus:outline-none min-w-[120px] text-xs sm:text-sm"
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
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Mode selector with two-button design */}
            <div className="flex items-center h-10 bg-slate-100 rounded-xl p-1 shadow-sm">
              <button
                onClick={() => setMode("ask")}
                className={`h-8 px-3 text-xs font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500 ${
                  mode === "ask"
                    ? "bg-blue-500 text-white shadow-sm"
                    : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                }`}
                aria-pressed={mode === "ask"}
              >
                Investigate
              </button>
              <button
                onClick={() => setMode("report")}
                className={`h-8 px-3 text-xs font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-green-500 ${
                  mode === "report"
                    ? "bg-green-500 text-white shadow-sm"
                    : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                }`}
                aria-pressed={mode === "report"}
              >
                Research Report
              </button>
            </div>

            {/* Effort level with modern toggle design - only show in Report mode */}
            {mode === "report" && (
              <div className="flex items-center h-10 bg-slate-100 rounded-xl p-1 shadow-sm">
                <button
                  onClick={() => setEffortLevel("quick")}
                  className={`h-8 px-3 text-xs font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-red-500 ${
                    effortLevel === "quick"
                      ? "bg-red-500 text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                  }`}
                  aria-pressed={effortLevel === "quick"}
                >
                  Quick
                </button>
                <button
                  onClick={() => setEffortLevel("standard")}
                  className={`h-8 px-3 text-xs font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-slate-500 ${
                    effortLevel === "standard"
                      ? "bg-slate-600 text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                  }`}
                  aria-pressed={effortLevel === "standard"}
                >
                  Standard
                </button>
                <button
                  onClick={() => setEffortLevel("high")}
                  className={`h-8 px-3 text-xs font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500 ${
                    effortLevel === "high"
                      ? "bg-blue-500 text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                  }`}
                  aria-pressed={effortLevel === "high"}
                >
                  High
                </button>
              </div>
            )}
          </div>

          {/* Enhanced submit button */}
          <button
            className={`group flex-shrink-0 flex items-center justify-center w-12 h-12 rounded-xl transition-all duration-200 shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 ${
              isSubmitting
                ? "bg-slate-400 cursor-not-allowed shadow-sm"
                : "bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 focus:ring-blue-500 hover:shadow-xl hover:scale-105 active:scale-95"
            }`}
            onClick={handleSubmit}
            disabled={isSubmitting}
            aria-label="Start Research"
            aria-disabled={isSubmitting}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className={`text-white transition-transform duration-200 ${!isSubmitting ? "group-hover:-translate-y-0.5" : ""}`}
            >
              <path d="M12 5V19" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              <path
                d="M5 12L12 5L19 12"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
