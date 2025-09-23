#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for specialized search functions:
- linkedin_search
- github_search
- academic_search

This script demonstrates how each specialized search function uses different
parameters for the Tavily API, and shows the results from each search.
"""

import os
import json
import argparse
import logging
from dotenv import load_dotenv
from src.utils import linkedin_search, github_search, academic_search

# Configure logging to show the search parameters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_linkedin_search(query, top_k=3, min_score=0.4):
    """Run a LinkedIn search and display results"""
    print("\n" + "="*80)
    print(f"LINKEDIN SEARCH: {query}")
    print("="*80)
    
    print("\nRunning LinkedIn search with the following constraints:")
    print(f"- Minimum score threshold: {min_score}")
    print(f"- Maximum results: {top_k}")
    print(f"- Domain restriction: linkedin.com")
    print(f"- Raw content: Only for highest scoring result\n")
    
    results = linkedin_search(
        query=query,
        include_raw_content=True,
        top_k=top_k,
        min_score=min_score
    )
    
    print(f"\nFound {len(results.get('results', []))} LinkedIn results with score >= {min_score}")
    
    for i, result in enumerate(results.get('results', [])):
        print(f"\nResult #{i+1}: score={result.get('score'):.4f}")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"Has raw content: {result.get('raw_content') is not None}")
        print("-" * 40)
        
    return results

def run_github_search(query, top_k=5, min_score=0.6):
    """Run a GitHub search and display results"""
    print("\n" + "="*80)
    print(f"GITHUB SEARCH: {query}")
    print("="*80)
    
    print("\nRunning GitHub search with the following constraints:")
    print(f"- Minimum score threshold: {min_score}")
    print(f"- Maximum results: {top_k}")
    print(f"- Domain restriction: github.com")
    print(f"- Score boost: 20% for actual repositories (/blob/ or /tree/ in URL)")
    print(f"- Raw content: Only kept for repositories with code content\n")
    
    results = github_search(
        query=query,
        include_raw_content=True,
        top_k=top_k,
        min_score=min_score
    )
    
    print(f"\nFound {len(results.get('results', []))} GitHub results with score >= {min_score}")
    
    for i, result in enumerate(results.get('results', [])):
        print(f"\nResult #{i+1}: score={result.get('score'):.4f}")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"Has raw content: {result.get('raw_content') is not None}")
        print("-" * 40)
        
    return results

def run_academic_search(query, top_k=5, min_score=0.65, recent_years=5):
    """Run an academic search and display results"""
    print("\n" + "="*80)
    print(f"ACADEMIC SEARCH: {query}")
    print("="*80)
    
    print("\nRunning Academic search with the following constraints:")
    print(f"- Minimum score threshold: {min_score}")
    print(f"- Maximum results: {top_k}")
    print(f"- Domain restriction: Academic domains (arxiv.org, scholar.google.com, etc.)")
    print(f"- Score boost: 15% for papers from the last {recent_years} years")
    print(f"- Score boost: Up to 30% for academic indicators (doi, abstract, etc.)")
    print(f"- Search depth: Advanced (for more comprehensive academic results)\n")
    
    results = academic_search(
        query=query,
        include_raw_content=True,
        top_k=top_k,
        min_score=min_score,
        recent_years=recent_years
    )
    
    print(f"\nFound {len(results.get('results', []))} academic results with score >= {min_score}")
    
    for i, result in enumerate(results.get('results', [])):
        print(f"\nResult #{i+1}: score={result.get('score'):.4f}")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"Has raw content: {result.get('raw_content') is not None}")
        print("-" * 40)
        
    return results

def main():
    """Main function to demonstrate all specialized search functions"""
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test specialized search functions')
    parser.add_argument('--type', '-t', choices=['linkedin', 'github', 'academic', 'all'], 
                        default='all', help='Type of search to run')
    parser.add_argument('--query', '-q', type=str, 
                        help='Search query to execute')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show full search results including raw content')
    args = parser.parse_args()
    
    # Set default queries for each search type if not provided
    linkedin_query = args.query or "frank wang salesforce, tell me about his background in details"
    github_query = args.query or "open source RAG framework python"
    academic_query = args.query or "large language models in healthcare research"
    
    # Run the requested search type(s)
    results = {}
    
    if args.type == 'all' or args.type == 'linkedin':
        results['linkedin'] = run_linkedin_search(linkedin_query)
        
    if args.type == 'all' or args.type == 'github':
        results['github'] = run_github_search(github_query)
        
    if args.type == 'all' or args.type == 'academic':
        results['academic'] = run_academic_search(academic_query)
    
    # If verbose mode, print the full JSON results
    if args.verbose:
        print("\n\nFULL SEARCH RESULTS (JSON):")
        print(json.dumps(results, indent=2))
    
    print("\nAll searches completed.")

if __name__ == "__main__":
    main() 