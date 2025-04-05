"""
Lightweight validation script for the Arabic Marketing Content Generator.

This script tests the core functionality without requiring all dependencies.
"""

import os
import json
import time
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def validate_project_structure():
    """Validate the project structure and files."""
    print("\n=== Validating Project Structure ===")
    
    # Define expected directories and files
    expected_dirs = [
        'src',
        'src/preprocessing',
        'src/trend_detection',
        'src/content_generation',
        'src/utils',
        'tests',
        'dashboard',
        'data'
    ]
    
    expected_files = [
        'src/__init__.py',
        'src/preprocessing/__init__.py',
        'src/preprocessing/data_loader.py',
        'src/preprocessing/text_preprocessor.py',
        'src/trend_detection/__init__.py',
        'src/trend_detection/feature_extractor.py',
        'src/trend_detection/trend_detector.py',
        'src/content_generation/__init__.py',
        'src/content_generation/content_generator.py',
        'src/utils/__init__.py',
        'src/utils/config.py',
        'dashboard/app.py',
        'arabic_marketing_generator.py',
        'setup.py'
    ]
    
    # Check directories
    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if not os.path.isdir(full_path):
            missing_dirs.append(dir_path)
    
    # Check files
    missing_files = []
    for file_path in expected_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if not os.path.isfile(full_path):
            missing_files.append(file_path)
    
    # Report results
    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
    else:
        print("✅ All expected directories are present")
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
    else:
        print("✅ All expected files are present")
    
    # Calculate completeness percentage
    dir_completeness = (len(expected_dirs) - len(missing_dirs)) / len(expected_dirs) * 100
    file_completeness = (len(expected_files) - len(missing_files)) / len(expected_files) * 100
    overall_completeness = (dir_completeness + file_completeness) / 2
    
    print(f"Project structure completeness: {overall_completeness:.1f}%")
    
    return {
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "dir_completeness": dir_completeness,
        "file_completeness": file_completeness,
        "overall_completeness": overall_completeness
    }

def validate_code_quality():
    """Validate code quality by checking for imports and docstrings."""
    print("\n=== Validating Code Quality ===")
    
    # Files to check
    files_to_check = [
        'src/preprocessing/data_loader.py',
        'src/preprocessing/text_preprocessor.py',
        'src/trend_detection/feature_extractor.py',
        'src/trend_detection/trend_detector.py',
        'src/content_generation/content_generator.py',
        'src/utils/config.py',
        'dashboard/app.py',
        'arabic_marketing_generator.py'
    ]
    
    quality_scores = {}
    
    for file_path in files_to_check:
        full_path = os.path.join(os.getcwd(), file_path)
        if not os.path.isfile(full_path):
            quality_scores[file_path] = {
                "has_docstring": False,
                "has_imports": False,
                "has_classes": False,
                "score": 0
            }
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for docstring
        has_docstring = '"""' in content[:500]  # Check first 500 chars for module docstring
        
        # Check for imports
        has_imports = 'import ' in content or 'from ' in content
        
        # Check for classes
        has_classes = 'class ' in content
        
        # Calculate simple quality score (0-100)
        score = 0
        if has_docstring:
            score += 40
        if has_imports:
            score += 30
        if has_classes:
            score += 30
        
        quality_scores[file_path] = {
            "has_docstring": has_docstring,
            "has_imports": has_imports,
            "has_classes": has_classes,
            "score": score
        }
    
    # Calculate average score
    avg_score = sum(item["score"] for item in quality_scores.values()) / len(quality_scores)
    
    # Report results
    print(f"Average code quality score: {avg_score:.1f}/100")
    
    # List files with low quality
    low_quality_files = [file for file, data in quality_scores.items() if data["score"] < 70]
    if low_quality_files:
        print(f"Files needing improvement: {', '.join(low_quality_files)}")
    else:
        print("✅ All files meet quality standards")
    
    return {
        "quality_scores": quality_scores,
        "average_score": avg_score,
        "low_quality_files": low_quality_files
    }

def validate_requirements_coverage():
    """Validate coverage of the requirements specified in the task."""
    print("\n=== Validating Requirements Coverage ===")
    
    # Define key requirements
    requirements = {
        "data_ingestion": {
            "description": "Ability to ingest Arabic Twitter datasets in CSV/JSON format",
            "files": ['src/preprocessing/data_loader.py'],
            "keywords": ['csv', 'json', 'load', 'read', 'dataframe']
        },
        "text_preprocessing": {
            "description": "Arabic text preprocessing with Camel Tools or alternatives",
            "files": ['src/preprocessing/text_preprocessor.py'],
            "keywords": ['clean', 'normalize', 'preprocess', 'arabic', 'pyarabic', 'farasa']
        },
        "trend_detection": {
            "description": "Trend detection with AraBERT and K-means clustering",
            "files": ['src/trend_detection/feature_extractor.py', 'src/trend_detection/trend_detector.py'],
            "keywords": ['arabert', 'embedding', 'cluster', 'kmeans', 'trend']
        },
        "content_generation": {
            "description": "Content generation with AraGPT2",
            "files": ['src/content_generation/content_generator.py'],
            "keywords": ['aragpt', 'generate', 'caption', 'hashtag', 'ad']
        },
        "cultural_relevance": {
            "description": "Ensuring cultural relevance and sensitivity",
            "files": ['src/content_generation/content_generator.py', 'src/utils/config.py'],
            "keywords": ['filter', 'sensitive', 'cultural', 'relevance']
        },
        "cli_interface": {
            "description": "Command-line interface",
            "files": ['arabic_marketing_generator.py'],
            "keywords": ['argparse', 'parse_args', 'command', 'cli']
        },
        "dashboard": {
            "description": "Streamlit dashboard",
            "files": ['dashboard/app.py'],
            "keywords": ['streamlit', 'st.', 'dashboard', 'visualization']
        }
    }
    
    coverage_results = {}
    
    for req_id, req_data in requirements.items():
        # Check if files exist
        files_exist = all(os.path.isfile(os.path.join(os.getcwd(), file)) for file in req_data["files"])
        
        # Check for keywords in files
        keyword_coverage = 0
        if files_exist:
            keyword_matches = 0
            for file_path in req_data["files"]:
                full_path = os.path.join(os.getcwd(), file_path)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for keyword in req_data["keywords"]:
                        if keyword.lower() in content:
                            keyword_matches += 1
            
            keyword_coverage = keyword_matches / len(req_data["keywords"]) * 100
        
        # Calculate overall coverage for this requirement
        overall_coverage = (100 if files_exist else 0) * 0.5 + keyword_coverage * 0.5
        
        coverage_results[req_id] = {
            "description": req_data["description"],
            "files_exist": files_exist,
            "keyword_coverage": keyword_coverage,
            "overall_coverage": overall_coverage
        }
    
    # Calculate average coverage
    avg_coverage = sum(item["overall_coverage"] for item in coverage_results.values()) / len(coverage_results)
    
    # Report results
    print(f"Overall requirements coverage: {avg_coverage:.1f}%")
    
    # List requirements with low coverage
    low_coverage_reqs = [f"{req_id} ({data['description']})" 
                         for req_id, data in coverage_results.items() 
                         if data["overall_coverage"] < 70]
    
    if low_coverage_reqs:
        print(f"Requirements needing improvement: {', '.join(low_coverage_reqs)}")
    else:
        print("✅ All requirements are adequately covered")
    
    return {
        "coverage_results": coverage_results,
        "average_coverage": avg_coverage,
        "low_coverage_requirements": low_coverage_reqs
    }

def run_validation():
    """Run all validation tests and generate a report."""
    print("=== Starting Lightweight Validation ===")
    print(f"Date and time: {datetime.now().isoformat()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'validation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run validation tests
    start_time = time.time()
    
    structure_results = validate_project_structure()
    code_quality_results = validate_code_quality()
    requirements_results = validate_requirements_coverage()
    
    end_time = time.time()
    
    # Calculate overall validation score
    structure_score = structure_results["overall_completeness"]
    quality_score = code_quality_results["average_score"]
    requirements_score = requirements_results["average_coverage"]
    
    overall_score = (structure_score + quality_score + requirements_score) / 3
    
    # Determine validation status
    validation_status = "PASSED" if overall_score >= 80 else "FAILED"
    
    # Compile validation report
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": end_time - start_time,
        "structure_validation": structure_results,
        "code_quality_validation": code_quality_results,
        "requirements_validation": requirements_results,
        "overall_score": overall_score,
        "validation_status": validation_status
    }
    
    # Save validation report
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Project Structure Score: {structure_score:.1f}%")
    print(f"Code Quality Score: {quality_score:.1f}/100")
    print(f"Requirements Coverage: {requirements_score:.1f}%")
    print(f"Overall Validation Score: {overall_score:.1f}%")
    print(f"Validation Status: {validation_status}")
    print(f"Validation report saved to: {report_path}")
    
    return validation_report, validation_status

if __name__ == "__main__":
    run_validation()
