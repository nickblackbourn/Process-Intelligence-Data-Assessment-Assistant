"""
Output Management Module for Process Mining Assessment Results

This module provides intelligent output file management with UX-focused organization,
addressing the common problem of file proliferation and poor naming conventions in 
data analysis tools.

Key Features:
- Organized directory structure with date-based folders
- Contextual file naming based on analyzed sources
- Automatic archiving of previous results
- Latest symlinks for easy access to newest results
- Support for multiple output formats (YAML, JSON)
- Cleanup utilities for maintenance

Directory Structure:
    results/
    ├── assessments/
    │   ├── 2025-08-08/           # Date-based organization
    │   │   ├── analysis1.yaml
    │   │   └── analysis2.yaml
    │   └── latest/               # Symlinks to newest files
    │       └── latest_assessment.yaml
    ├── reports/                  # Future: Human-readable reports
    ├── sql/                      # Future: Generated SQL queries
    └── archives/                 # Archived files when overwritten

Author: Process Intelligence Team
Version: 1.0.0 - Initial UX-focused output management
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output files with intelligent organization and naming."""
    
    def __init__(self, base_dir: str = "results"):
        """Initialize the OutputManager.
        
        Args:
            base_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_dir)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_time = datetime.now().strftime("%H-%M-%S")
        
        # Create directory structure
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """Create the organized directory structure."""
        dirs_to_create = [
            self.base_dir / "assessments" / self.current_date,
            self.base_dir / "assessments" / "latest",
            self.base_dir / "reports" / self.current_date,
            self.base_dir / "reports" / "latest", 
            self.base_dir / "sql" / self.current_date,
            self.base_dir / "sql" / "latest",
            self.base_dir / "archives"
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ensured output directory structure in {self.base_dir}")
    
    def generate_contextual_filename(
        self, 
        data_sources: List[str], 
        analysis_type: str = "assessment",
        custom_name: Optional[str] = None,
        file_extension: str = "yaml"
    ) -> str:
        """Generate a contextual filename based on analysis details.
        
        Args:
            data_sources: List of data source files analyzed
            analysis_type: Type of analysis performed
            custom_name: Optional custom base name
            file_extension: File extension (without dot)
            
        Returns:
            Intelligently generated filename
        """
        if custom_name:
            base_name = custom_name
        else:
            # Generate context-aware name
            if len(data_sources) == 1:
                source_file = Path(data_sources[0]).stem
                # Clean up filename (remove version numbers, special chars)
                source_file = self._clean_filename(source_file)
                base_name = f"{analysis_type}_{source_file}"
            elif len(data_sources) > 1:
                # Multiple sources
                base_name = f"{analysis_type}_{len(data_sources)}sources"
            else:
                base_name = analysis_type
        
        # Add timestamp for uniqueness
        filename = f"{base_name}_{self.current_date}_{self.current_time}.{file_extension}"
        return filename
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for better readability.
        
        Args:
            filename: Original filename to clean
            
        Returns:
            Cleaned filename
        """
        # Remove common patterns that clutter filenames
        import re
        
        # Remove version patterns like _v9.56_, _version_9.56_
        filename = re.sub(r'_v\d+\.\d+_?', '', filename, flags=re.IGNORECASE)
        filename = re.sub(r'_version_\d+\.\d+_?', '', filename, flags=re.IGNORECASE)
        
        # Remove excessive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Remove trailing underscores
        filename = filename.strip('_')
        
        # Limit length
        if len(filename) > 30:
            filename = filename[:30]
        
        return filename
    
    def get_output_path(
        self,
        data_sources: List[str],
        analysis_type: str = "assessment", 
        custom_name: Optional[str] = None,
        output_format: str = "yaml",
        use_date_folder: bool = True
    ) -> Path:
        """Get the full output path for a file.
        
        Args:
            data_sources: List of data source files
            analysis_type: Type of analysis
            custom_name: Optional custom filename base
            output_format: Output file format
            use_date_folder: Whether to use date-based folder organization
            
        Returns:
            Full path where file should be saved
        """
        filename = self.generate_contextual_filename(
            data_sources, analysis_type, custom_name, output_format
        )
        
        if use_date_folder:
            if analysis_type == "assessment":
                output_dir = self.base_dir / "assessments" / self.current_date
            elif analysis_type == "report":
                output_dir = self.base_dir / "reports" / self.current_date  
            elif analysis_type == "sql":
                output_dir = self.base_dir / "sql" / self.current_date
            else:
                output_dir = self.base_dir / "assessments" / self.current_date
        else:
            output_dir = self.base_dir / "assessments"
        
        return output_dir / filename
    
    def save_assessment(
        self,
        assessment_data: Dict[str, Any],
        data_sources: List[str],
        custom_name: Optional[str] = None,
        output_format: str = "yaml",
        keep_history: bool = True
    ) -> Path:
        """Save assessment data with intelligent file management.
        
        Args:
            assessment_data: Assessment results to save
            data_sources: List of data sources analyzed
            custom_name: Optional custom filename
            output_format: Output format (yaml, json)
            keep_history: Whether to keep previous results
            
        Returns:
            Path where file was saved
        """
        output_path = self.get_output_path(
            data_sources, "assessment", custom_name, output_format
        )
        
        # Archive previous file if it exists and keep_history is True
        if output_path.exists() and keep_history:
            self.archive_file(output_path)
        
        # Save the new file
        if output_format.lower() == "yaml":
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(assessment_data, f, default_flow_style=False, 
                              sort_keys=False, allow_unicode=True)
        elif output_format.lower() == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(assessment_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Update latest symlink
        self.update_latest_link(output_path, "assessment")
        
        logger.info(f"Assessment saved to {output_path}")
        return output_path
    
    def archive_file(self, file_path: Path) -> Path:
        """Archive an existing file to prevent overwrite.
        
        Args:
            file_path: Path to file to archive
            
        Returns:
            Path where file was archived
        """
        if not file_path.exists():
            return file_path
        
        # Create archive filename with timestamp
        archive_name = f"{file_path.stem}_archived_{self.current_date}_{self.current_time}{file_path.suffix}"
        archive_path = self.base_dir / "archives" / archive_name
        
        # Move to archive
        shutil.move(str(file_path), str(archive_path))
        logger.info(f"Archived previous file to {archive_path}")
        
        return archive_path
    
    def update_latest_link(self, file_path: Path, file_type: str) -> None:
        """Update the 'latest' symlink to point to newest file.
        
        Args:
            file_path: Path to the newest file
            file_type: Type of file (assessment, report, sql)
        """
        latest_dir = self.base_dir / f"{file_type}s" / "latest"
        latest_file = latest_dir / f"latest_{file_type}{file_path.suffix}"
        
        # Remove existing symlink if it exists
        if latest_file.exists() or latest_file.is_symlink():
            latest_file.unlink()
        
        try:
            # Create relative symlink
            relative_path = os.path.relpath(file_path, latest_file.parent)
            latest_file.symlink_to(relative_path)
            logger.debug(f"Updated latest {file_type} link: {latest_file}")
        except OSError:
            # Symlinks might not work on all systems, copy instead
            shutil.copy2(file_path, latest_file)
            logger.debug(f"Copied to latest {file_type}: {latest_file}")
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """Clean up old files beyond retention period.
        
        Args:
            days_to_keep: Number of days to keep files
            
        Returns:
            Number of files cleaned up
        """
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        cleaned_count = 0
        
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                # Don't clean latest directory or recent archives
                if "latest" not in file_path.parts and file_path.stat().st_mtime < cutoff_date:
                    file_path.unlink()
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old files")
        return cleaned_count
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all results in the output directory.
        
        Returns:
            Summary of files and organization
        """
        summary = {
            "base_directory": str(self.base_dir),
            "total_files": 0,
            "by_type": {},
            "by_date": {},
            "latest_files": {}
        }
        
        for file_path in self.base_dir.rglob("*.yaml"):
            if file_path.is_file():
                summary["total_files"] += 1
                
                # Categorize by parent directory
                parent = file_path.parent.name
                if parent not in summary["by_type"]:
                    summary["by_type"][parent] = 0
                summary["by_type"][parent] += 1
                
                # Check if it's in latest
                if "latest" in file_path.parts:
                    file_type = file_path.parent.parent.name
                    summary["latest_files"][file_type] = str(file_path)
        
        return summary
