"""
Filter papers based on related works quality and citation count.

This script processes pipeline outputs and filters papers to ensure:
1. Papers have a related works section
2. Related works section is not too short or too long
3. Papers have at least a minimum number of citations

Usage:
    python filter_quality_papers.py --input-folder outputs/20251015_143311/ \
                                     --min-citations 5 \
                                     --min-rw-length 200 \
                                     --max-rw-length 10000
"""

import argparse
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperQualityFilter:
    """Filter papers based on quality criteria."""
    
    def __init__(
        self, 
        input_folder: str,
        min_citations: int = 5,
        min_rw_length: int = 200,
        max_rw_length: int = 10000,
        citations_folder: Optional[str] = None,
        related_works_folder: Optional[str] = None
    ):
        """
        Initialize the quality filter.
        
        Args:
            input_folder: Path to the folder containing paper_content.csv
            min_citations: Minimum number of citations required
            min_rw_length: Minimum length of related works section (characters)
            max_rw_length: Maximum length of related works section (characters)
            citations_folder: Path to folder with citation CSVs (default: ../citations/)
            related_works_folder: Path to folder with related works CSVs (default: ../related_works/)
        """
        self.input_folder = Path(input_folder)
        self.min_citations = min_citations
        self.min_rw_length = min_rw_length
        self.max_rw_length = max_rw_length
        
        # Set up paths
        self.paper_content_path = self.input_folder / "paper_content.csv"
        self.papers_path = self.input_folder / "papers.csv"
        
        # Use provided paths or default to parent folders
        if citations_folder:
            self.citations_folder = Path(citations_folder)
        else:
            self.citations_folder = self.input_folder.parent / "citations"
            
        if related_works_folder:
            self.related_works_folder = Path(related_works_folder)
        else:
            self.related_works_folder = self.input_folder.parent / "related_works"
        
        # Validate paths
        if not self.paper_content_path.exists():
            raise FileNotFoundError(f"paper_content.csv not found at {self.paper_content_path}")
    
    def count_citations(self, arxiv_id: str) -> int:
        """
        Count citations for a paper by reading its citation CSV.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "2509.11056v1")
            
        Returns:
            Number of citations found
        """
        citation_file = self.citations_folder / f"{arxiv_id}.csv"
        
        if not citation_file.exists():
            logger.debug(f"No citation file found for {arxiv_id}")
            return 0
        
        try:
            citations_df = pd.read_csv(citation_file)
            return len(citations_df)
        except Exception as e:
            logger.warning(f"Error reading citation file for {arxiv_id}: {e}")
            return 0
    
    def validate_related_works(self, row: pd.Series) -> bool:
        """
        Check if related works section meets quality criteria.
        
        Args:
            row: DataFrame row with paper data
            
        Returns:
            True if related works section is valid
        """
        # Check if related works section exists
        if pd.isna(row.get("related_works_section")) or not row.get("related_works_section"):
            logger.debug(f"Paper {row.get('arxiv_id')} has no related works section")
            return False
        
        # Check related works length
        rw_length = row.get("related_works_length", 0)
        
        # If length column doesn't exist or is NaN, calculate from text
        if pd.isna(rw_length):
            rw_text = str(row.get("related_works_section", ""))
            rw_length = len(rw_text)
        
        if rw_length < self.min_rw_length:
            logger.debug(
                f"Paper {row.get('arxiv_id')} related works too short: {rw_length} chars"
            )
            return False
        
        if rw_length > self.max_rw_length:
            logger.debug(
                f"Paper {row.get('arxiv_id')} related works too long: {rw_length} chars"
            )
            return False
        
        return True
    
    def filter_papers(self) -> pd.DataFrame:
        """
        Filter papers based on all quality criteria.
        
        Returns:
            Filtered DataFrame with quality papers
        """
        logger.info(f"Reading papers from {self.paper_content_path}")
        df = pd.read_csv(self.paper_content_path)
        
        initial_count = len(df)
        logger.info(f"Total papers in input: {initial_count}")
        
        # Track filtering statistics
        stats = {
            "no_related_works": 0,
            "rw_too_short": 0,
            "rw_too_long": 0,
            "insufficient_citations": 0,
            "passed": 0
        }
        
        # Lists to store filtered data
        filtered_rows = []
        
        for idx, row in df.iterrows():
            arxiv_id = row.get("arxiv_id")
            
            # Check related works section
            if pd.isna(row.get("related_works_section")) or not row.get("related_works_section"):
                stats["no_related_works"] += 1
                continue
            
            # Check related works length
            rw_length = row.get("related_works_length", 0)
            if pd.isna(rw_length):
                rw_text = str(row.get("related_works_section", ""))
                rw_length = len(rw_text)
            
            if rw_length < self.min_rw_length:
                stats["rw_too_short"] += 1
                continue
            
            if rw_length > self.max_rw_length:
                stats["rw_too_long"] += 1
                continue
            
            # Check citation count
            citation_count = self.count_citations(arxiv_id)
            if citation_count < self.min_citations:
                stats["insufficient_citations"] += 1
                logger.debug(
                    f"Paper {arxiv_id} has only {citation_count} citations "
                    f"(minimum: {self.min_citations})"
                )
                continue
            
            # Paper passed all filters
            stats["passed"] += 1
            filtered_rows.append(row)
            
            if stats["passed"] % 100 == 0:
                logger.info(f"Processed {idx + 1}/{initial_count} papers, {stats['passed']} passed")
        
        # Create filtered dataframe
        filtered_df = pd.DataFrame(filtered_rows)
        
        # Log statistics
        logger.info("\n" + "=" * 60)
        logger.info("FILTERING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total papers processed: {initial_count}")
        logger.info(f"Papers with no related works: {stats['no_related_works']}")
        logger.info(f"Papers with related works too short: {stats['rw_too_short']}")
        logger.info(f"Papers with related works too long: {stats['rw_too_long']}")
        logger.info(f"Papers with insufficient citations: {stats['insufficient_citations']}")
        logger.info(f"Papers that passed filters: {stats['passed']}")
        logger.info(f"Pass rate: {stats['passed'] / initial_count * 100:.2f}%")
        logger.info("=" * 60)
        
        return filtered_df
    
    def filter_and_save(self, output_suffix: str = "_filtered"):
        """
        Filter papers and save results.
        
        Args:
            output_suffix: Suffix to add to output filenames
        """
        # Filter papers
        filtered_df = self.filter_papers()
        
        # Save filtered paper_content.csv
        paper_content_output = self.input_folder / f"paper_content{output_suffix}.csv"
        filtered_df.to_csv(paper_content_output, index=False)
        logger.info(f"Saved filtered paper content to {paper_content_output}")
        
        # Also filter papers.csv if it exists
        if self.papers_path.exists():
            logger.info(f"Filtering papers.csv based on filtered arxiv_ids...")
            papers_df = pd.read_csv(self.papers_path)
            
            # Get list of filtered arxiv_ids
            filtered_ids = set(filtered_df["arxiv_id"].values)
            
            # Filter papers.csv
            filtered_papers_df = papers_df[papers_df["arxiv_id"].isin(filtered_ids)]
            
            papers_output = self.input_folder / f"papers{output_suffix}.csv"
            filtered_papers_df.to_csv(papers_output, index=True)
            logger.info(f"Saved filtered papers to {papers_output}")
            logger.info(f"Papers in papers.csv: {len(papers_df)} -> {len(filtered_papers_df)}")
        
        return filtered_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter papers based on related works quality and citation count"
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing paper_content.csv (e.g., outputs/20251015_143311/)"
    )
    
    parser.add_argument(
        "--min-citations",
        type=int,
        default=5,
        help="Minimum number of citations required (default: 5)"
    )
    
    parser.add_argument(
        "--min-rw-length",
        type=int,
        default=200,
        help="Minimum length of related works section in characters (default: 200)"
    )
    
    parser.add_argument(
        "--max-rw-length",
        type=int,
        default=10000,
        help="Maximum length of related works section in characters (default: 10000)"
    )
    
    parser.add_argument(
        "--citations-folder",
        type=str,
        default=None,
        help="Path to citations folder (default: ../citations/ relative to input folder)"
    )
    
    parser.add_argument(
        "--related-works-folder",
        type=str,
        default=None,
        help="Path to related works folder (default: ../related_works/ relative to input folder)"
    )
    
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_filtered",
        help="Suffix to add to output filenames (default: _filtered)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting paper quality filtering...")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Minimum citations: {args.min_citations}")
    logger.info(f"Related works length range: {args.min_rw_length} - {args.max_rw_length} chars")
    
    # Create filter instance
    filter_obj = PaperQualityFilter(
        input_folder=args.input_folder,
        min_citations=args.min_citations,
        min_rw_length=args.min_rw_length,
        max_rw_length=args.max_rw_length,
        citations_folder=args.citations_folder,
        related_works_folder=args.related_works_folder
    )
    
    # Filter and save
    filter_obj.filter_and_save(output_suffix=args.output_suffix)
    
    logger.info("✅ Filtering complete!")


if __name__ == "__main__":
    main()



