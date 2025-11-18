#!/usr/bin/env python3
"""
Script to fix incomplete S2AG database by adding missing papers, paper_authors, and citations data
Works with absolute paths to avoid directory-related issues
"""

import gzip
import json
import sqlite3
import logging
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fix_database.log')
    ]
)
logger = logging.getLogger(__name__)

class DatabaseFixer:
    """Fix incomplete database by adding missing data"""
    
    def __init__(self, db_path: str, data_dir: str, batch_size: int = 1000):
        """
        Args:
            db_path: Path to the database file
            data_dir: Directory containing the data folders (authors/, papers/, etc.)
            batch_size: Number of records to insert in each batch
        """
        self.db_path = Path(db_path).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.conn = None
        self.error_counts = defaultdict(int)
        self.processed_counts = defaultdict(int)
        self.start_time = None
        
        # Validate paths
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def connect(self):
        """Connect to database"""
        logger.info("Connecting to database...")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=1000000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        logger.info("Connected successfully")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get current row counts for all tables"""
        cursor = self.conn.cursor()
        counts = {}
        tables = ['authors', 'publication_venues', 'papers', 'paper_authors', 'citations']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        
        return counts
    
    def safe_json_loads(self, line: str, line_num: int, file_path: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON with error handling"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            self.error_counts['json_decode'] += 1
            if self.error_counts['json_decode'] <= 10:  # Only log first 10 errors
                logger.warning(f"JSON decode error in {file_path}:{line_num}: {e}")
            return None
    
    def safe_json_dumps(self, obj: Any) -> Optional[str]:
        """Safely serialize to JSON"""
        if obj is None:
            return None
        try:
            return json.dumps(obj)
        except (TypeError, ValueError) as e:
            self.error_counts['json_serialize'] += 1
            return None
    
    def read_jsonl_gz_files(self, subfolder: str, max_files: Optional[int] = None):
        """
        Generator to read all .jsonl.gz files in a subfolder
        
        Args:
            subfolder: Subfolder name (e.g., 'papers', 'citations')
            max_files: Maximum number of files to process (None for all)
        """
        pattern = f"part_*.jsonl.gz"
        folder_path = self.data_dir / subfolder
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return
        
        files = sorted(folder_path.glob(pattern))
        
        if not files:
            logger.error(f"No files found in {folder_path} matching {pattern}")
            return
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} files in {subfolder}/")
        
        for file_idx, file_path in enumerate(files, 1):
            logger.info(f"Processing file {file_idx}/{len(files)}: {file_path.name}")
            
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        if line.strip():
                            record = self.safe_json_loads(line, line_num, str(file_path))
                            if record:
                                yield record, str(file_path), line_num
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                self.error_counts['file_read'] += 1
                continue
    
    def process_papers(self, max_files: Optional[int] = None):
        """Process papers and paper_authors data"""
        logger.info("="*60)
        logger.info("Processing papers and paper_authors...")
        logger.info("="*60)
        
        cursor = self.conn.cursor()
        paper_batch = []
        author_batch = []
        
        paper_count = 0
        author_rel_count = 0
        
        for record, file_path, line_num in self.read_jsonl_gz_files("papers", max_files):
            try:
                corpus_id = record.get('corpusid')
                if not corpus_id:
                    self.error_counts['paper_missing_corpusid'] += 1
                    continue
                
                paper_id = str(corpus_id)
                
                # Extract paper data
                paper_data = (
                    paper_id,
                    corpus_id,
                    record.get('title'),
                    None,  # abstract - not in current data
                    record.get('year'),
                    record.get('publicationdate'),
                    record.get('venue', ''),
                    record.get('publicationvenueid'),
                    record.get('referencecount', 0),
                    record.get('citationcount', 0),
                    record.get('influentialcitationcount', 0),
                    record.get('isopenaccess', False),
                    None,  # fields_of_study
                    self.safe_json_dumps(record.get('s2fieldsofstudy')),
                    self.safe_json_dumps(record.get('publicationtypes')),
                    self.safe_json_dumps(record.get('externalids')),
                    self.safe_json_dumps(record.get('journal')),
                    record.get('url')
                )
                
                if not paper_data[2]:  # title required
                    self.error_counts['paper_missing_title'] += 1
                    continue
                
                paper_batch.append(paper_data)
                paper_count += 1
                
                # Process authors for this paper
                authors = record.get('authors', [])
                for pos, author in enumerate(authors, 1):
                    if isinstance(author, dict) and author.get('authorId'):
                        author_rel_data = (
                            paper_id,
                            author.get('authorId'),
                            pos,
                            False  # is_corresponding_author
                        )
                        author_batch.append(author_rel_data)
                        author_rel_count += 1
                
                # Insert batches when they reach size limit
                if len(paper_batch) >= self.batch_size:
                    self._insert_paper_batch(cursor, paper_batch)
                    paper_batch = []
                
                if len(author_batch) >= self.batch_size:
                    self._insert_paper_author_batch(cursor, author_batch)
                    author_batch = []
                
                # Periodic progress report
                if paper_count % 10000 == 0:
                    logger.info(f"Processed {paper_count:,} papers, {author_rel_count:,} author relationships")
                    
            except Exception as e:
                self.error_counts['paper_processing'] += 1
                if self.error_counts['paper_processing'] <= 10:
                    logger.warning(f"Error processing paper from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batches
        if paper_batch:
            self._insert_paper_batch(cursor, paper_batch)
        if author_batch:
            self._insert_paper_author_batch(cursor, author_batch)
        
        self.conn.commit()
        logger.info(f"✓ Processed {self.processed_counts['papers']:,} papers")
        logger.info(f"✓ Processed {self.processed_counts['paper_authors']:,} paper-author relationships")
    
    def _insert_paper_batch(self, cursor, batch: List[tuple]):
        """Insert batch of papers"""
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO papers 
                (paper_id, corpus_id, title, abstract, year, publication_date, venue,
                 publication_venue_id, reference_count, citation_count, influential_citation_count,
                 is_open_access, fields_of_study, s2_fields_of_study, publication_types,
                 external_ids, journal_info, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            self.processed_counts['papers'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting paper batch: {e}")
            self.error_counts['paper_insert'] += len(batch)
    
    def _insert_paper_author_batch(self, cursor, batch: List[tuple]):
        """Insert batch of paper-author relationships"""
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO paper_authors 
                (paper_id, author_id, author_position, is_corresponding_author)
                VALUES (?, ?, ?, ?)
            """, batch)
            self.processed_counts['paper_authors'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting paper-author batch: {e}")
            self.error_counts['paper_author_insert'] += len(batch)
    
    def process_citations(self, max_files: Optional[int] = None):
        """Process citations data"""
        logger.info("="*60)
        logger.info("Processing citations...")
        logger.info("="*60)
        
        cursor = self.conn.cursor()
        batch = []
        citation_count = 0
        
        for record, file_path, line_num in self.read_jsonl_gz_files("citations", max_files):
            try:
                citing_corpus_id = record.get('citingcorpusid')
                cited_corpus_id = record.get('citedcorpusid')
                
                if not citing_corpus_id:
                    self.error_counts['citation_missing_citing'] += 1
                    continue
                
                citation_data = (
                    record.get('citationid'),
                    str(citing_corpus_id),
                    str(cited_corpus_id) if cited_corpus_id else None,
                    self.safe_json_dumps(record.get('contexts')),
                    self.safe_json_dumps(record.get('intents')),
                    record.get('isinfluential', False)
                )
                
                batch.append(citation_data)
                citation_count += 1
                
                if len(batch) >= self.batch_size:
                    self._insert_citation_batch(cursor, batch)
                    batch = []
                
                # Periodic progress report
                if citation_count % 100000 == 0:
                    logger.info(f"Processed {citation_count:,} citations")
                    
            except Exception as e:
                self.error_counts['citation_processing'] += 1
                if self.error_counts['citation_processing'] <= 10:
                    logger.warning(f"Error processing citation from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batch
        if batch:
            self._insert_citation_batch(cursor, batch)
        
        self.conn.commit()
        logger.info(f"✓ Processed {self.processed_counts['citations']:,} citations")
    
    def _insert_citation_batch(self, cursor, batch: List[tuple]):
        """Insert batch of citations"""
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO citations 
                (citation_id, citing_paper_id, cited_paper_id, contexts, intents, is_influential)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            self.processed_counts['citations'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting citation batch: {e}")
            self.error_counts['citation_insert'] += len(batch)
    
    def create_indexes(self):
        """Create indexes if they don't exist"""
        logger.info("Creating/verifying indexes...")
        
        cursor = self.conn.cursor()
        
        indexes = [
            # Papers indexes
            "CREATE INDEX IF NOT EXISTS idx_papers_corpus_id ON papers(corpus_id)",
            "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)",
            "CREATE INDEX IF NOT EXISTS idx_papers_venue_id ON papers(publication_venue_id)",
            "CREATE INDEX IF NOT EXISTS idx_papers_citation_count ON papers(citation_count)",
            
            # Paper-authors indexes
            "CREATE INDEX IF NOT EXISTS idx_paper_authors_paper ON paper_authors(paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id)",
            
            # Citations indexes
            "CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_citations_influential ON citations(is_influential)",
        ]
        
        for idx_sql in indexes:
            try:
                cursor.execute(idx_sql)
                logger.info(f"  {idx_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Error creating index: {e}")
        
        self.conn.commit()
        logger.info("Indexes created/verified")
    
    def optimize_database(self):
        """Optimize database after bulk inserts"""
        logger.info("Optimizing database...")
        
        try:
            cursor = self.conn.cursor()
            
            logger.info("  Running ANALYZE...")
            cursor.execute("ANALYZE")
            
            logger.info("  Running VACUUM...")
            cursor.execute("VACUUM")
            
            self.conn.commit()
            logger.info("Database optimized")
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def print_summary(self, initial_counts: Dict[str, int]):
        """Print summary of what was done"""
        final_counts = self.get_table_counts()
        
        logger.info("="*60)
        logger.info("DATABASE FIX SUMMARY")
        logger.info("="*60)
        
        logger.info("\nTable changes:")
        for table in ['authors', 'publication_venues', 'papers', 'paper_authors', 'citations']:
            initial = initial_counts[table]
            final = final_counts[table]
            added = final - initial
            logger.info(f"  {table:20}: {initial:>12,} → {final:>12,} (+{added:,})")
        
        if self.error_counts:
            logger.info("\nErrors encountered:")
            for error_type, count in sorted(self.error_counts.items()):
                logger.info(f"  {error_type}: {count:,}")
        
        # Calculate total time
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            
            total_records = sum(self.processed_counts.values())
            if total_time > 0:
                rate = total_records / total_time
                logger.info(f"Processing rate: {rate:.0f} records/second")
        
        # Database size
        db_size = self.db_path.stat().st_size / (1024**3)  # GB
        logger.info(f"\nDatabase size: {db_size:.2f} GB")
        
        logger.info("="*60)
    
    def fix_database(self, process_papers: bool = True, process_citations: bool = True, 
                     max_paper_files: Optional[int] = None, 
                     max_citation_files: Optional[int] = None,
                     skip_optimization: bool = False):
        """
        Main method to fix the database
        
        Args:
            process_papers: Whether to process papers data
            process_citations: Whether to process citations data
            max_paper_files: Max paper files to process (None for all)
            max_citation_files: Max citation files to process (None for all)
            skip_optimization: Skip VACUUM (useful for large databases)
        """
        self.start_time = time.time()
        
        try:
            self.connect()
            
            # Get initial counts
            logger.info("\nInitial table counts:")
            initial_counts = self.get_table_counts()
            for table, count in initial_counts.items():
                logger.info(f"  {table}: {count:,}")
            
            # Process data
            if process_papers:
                self.process_papers(max_files=max_paper_files)
            
            if process_citations:
                self.process_citations(max_files=max_citation_files)
            
            # Create indexes
            self.create_indexes()
            
            # Optimize database
            if not skip_optimization:
                self.optimize_database()
            else:
                logger.info("Skipping optimization (VACUUM) as requested")
            
            # Print summary
            self.print_summary(initial_counts)
            
            logger.info("\n✅ Database fix completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.error("\n❌ Fix interrupted by user")
            return False
        except Exception as e:
            logger.error(f"\n❌ Fix failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Fix incomplete S2AG database by adding missing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on small subset (1 file each)
  python fix_database.py --db-path /path/to/s2ag_database.db --data-dir . --max-paper-files 1 --max-citation-files 1
  
  # Fix with limited files (good for testing)
  python fix_database.py --db-path /path/to/s2ag_database.db --data-dir . --max-paper-files 10
  
  # Full fix (process all data)
  python fix_database.py --db-path /path/to/s2ag_database.db --data-dir .
  
  # Only add papers (skip citations)
  python fix_database.py --db-path /path/to/s2ag_database.db --data-dir . --skip-citations
        """
    )
    
    parser.add_argument("--db-path", required=True,
                       help="Path to the database file to fix")
    parser.add_argument("--data-dir", required=True,
                       help="Directory containing data folders (authors/, papers/, etc.)")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for inserts (default: 1000)")
    parser.add_argument("--max-paper-files", type=int, default=None,
                       help="Max paper files to process (for testing)")
    parser.add_argument("--max-citation-files", type=int, default=None,
                       help="Max citation files to process (for testing)")
    parser.add_argument("--skip-papers", action="store_true",
                       help="Skip processing papers")
    parser.add_argument("--skip-citations", action="store_true",
                       help="Skip processing citations")
    parser.add_argument("--skip-optimization", action="store_true",
                       help="Skip VACUUM (faster, but database may be larger)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_papers and args.skip_citations:
        logger.error("Cannot skip both papers and citations!")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("S2AG DATABASE FIXER")
    logger.info("="*60)
    
    # Create fixer and run
    fixer = DatabaseFixer(
        db_path=args.db_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    success = fixer.fix_database(
        process_papers=not args.skip_papers,
        process_citations=not args.skip_citations,
        max_paper_files=args.max_paper_files,
        max_citation_files=args.max_citation_files,
        skip_optimization=args.skip_optimization
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
