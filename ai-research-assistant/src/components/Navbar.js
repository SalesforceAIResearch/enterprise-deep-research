import React from 'react';

const Navbar = ({
    isResearching,
    onShowProgress,
    onShowReport,
    hasReport,
    isProgressOpen,
    isReportOpen
}) => {
    return (
        <nav className="bg-gradient-to-r from-white via-blue-50/30 to-white border-b-2 border-slate-200 shadow-lg backdrop-blur-sm sticky top-0 z-50">
            <div className="w-full px-6 py-3 flex items-center justify-between">
                {/* Left: Logo and Title */}
                <div className="flex items-center gap-4 flex-shrink-0">
                    <img
                        src="/sfr_logo.jpeg"
                        alt="Salesforce Research"
                        className="h-10 w-auto object-contain hover:scale-105 transition-transform"
                    />
                    <div className="border-l-2 border-slate-300 pl-4">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-[#032d60] via-[#0176d3] to-[#032d60] bg-clip-text text-transparent whitespace-nowrap tracking-tight leading-tight">
                            Enterprise Deep Research: Steerable Deep Research for Enterprise
                        </h1>
                    </div>
                </div>

                {/* Right side: Panel Toggles and Actions */}
                <div className="flex items-center gap-3 flex-shrink-0">
                    {/* Panel Toggles (only show during research) */}
                    {isResearching && (
                        <div className="flex items-center gap-2 mr-2">
                            {/* Progress Panel Toggle */}
                            <button
                                onClick={onShowProgress}
                                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-200 ${isProgressOpen
                                    ? 'bg-[#0176d3] text-white shadow-lg shadow-[#0176d3]/30 scale-105 border border-[#0176d3]'
                                    : 'bg-white text-[#032d60] hover:bg-[#e3f3ff] border border-slate-200 hover:border-[#0176d3] shadow-sm hover:shadow-md hover:scale-105'
                                    }`}
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                </svg>
                                <span>Progress</span>
                            </button>

                            {/* Report Panel Toggle */}
                            {hasReport && (
                                <button
                                    onClick={onShowReport}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm transition-all duration-200 ${isReportOpen
                                        ? 'bg-[#04844b] text-white shadow-lg shadow-[#04844b]/30 scale-105 border border-[#04844b]'
                                        : 'bg-white text-[#032d60] hover:bg-[#e6f5ef] border border-slate-200 hover:border-[#04844b] shadow-sm hover:shadow-md hover:scale-105'
                                        }`}
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    <span>Report</span>
                                </button>
                            )}
                        </div>
                    )}

                    {/* Vertical Divider */}
                    <div className="h-8 w-px bg-slate-300 mx-2"></div>

                    {/* GitHub Link */}
                    <a
                        href="https://github.com/SalesforceAIResearch/enterprise-deep-research"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-black hover:bg-gray-800 text-white rounded-lg transition-all duration-200 text-xs font-semibold shadow-md hover:shadow-lg hover:scale-105 border border-gray-900"
                        title="View on GitHub"
                    >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                            <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                        </svg>
                        <span>GitHub</span>
                    </a>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
