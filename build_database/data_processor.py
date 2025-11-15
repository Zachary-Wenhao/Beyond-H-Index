#!/usr/bin/env python3
"""
Robust data processing pipeline for S2AG data
Handles errors gracefully, processes in batches, and provides progress tracking
"""

import gzip
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple
from collections import defaultdict
import time
from datetime import datetime
import traceback

from database_schema import DatabaseSchema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Robust data processor with error handling and batch processing"""
    
    def __init__(self, db_path: str = "s2ag_database.db", batch_size: int = 1000):
        self.db_path = db_path
        self.batch_size = batch_size
        self.schema = DatabaseSchema(db_path)
        self.error_counts = defaultdict(int)
        self.processed_counts = defaultdict(int)
        self.start_time = None
        
    def safe_json_loads(self, line: str, line_num: int, file_path: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON with error handling"""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            self.error_counts['json_decode'] += 1
            logger.warning(f"JSON decode error in {file_path}:{line_num}: {e}")
            return None
        except Exception as e:
            self.error_counts['json_other'] += 1
            logger.warning(f"Unexpected JSON error in {file_path}:{line_num}: {e}")
            return None
    
    def safe_json_dumps(self, obj: Any) -> Optional[str]:
        """Safely serialize to JSON"""
        if obj is None:
            return None
        try:
            return json.dumps(obj)
        except (TypeError, ValueError) as e:
            self.error_counts['json_serialize'] += 1
            logger.warning(f"JSON serialization error: {e}")
            return None
    
    def read_jsonl_gz_files(self, file_pattern: str) -> Generator[Tuple[Dict[str, Any], str, int], None, None]:
        """Generator to read all files matching pattern"""
        files = list(Path('.').glob(file_pattern))
        files.sort()  # Process in consistent order
        
        logger.info(f"Found {len(files)} files matching pattern: {file_pattern}")
        
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        if line.strip():  # Skip empty lines
                            record = self.safe_json_loads(line, line_num, str(file_path))
                            if record:
                                yield record, str(file_path), line_num
                                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                self.error_counts['file_read'] += 1
                continue
    
    def process_publication_venues(self):
        """Process publication venues data"""
        logger.info("Processing publication venues...")
        
        cursor = self.schema.conn.cursor()
        batch = []
        
        for record, file_path, line_num in self.read_jsonl_gz_files("publication-venues/part_*.jsonl.gz"):
            try:
                venue_data = (
                    record.get('id'),
                    record.get('name'),
                    record.get('type'),
                    self.safe_json_dumps(record.get('alternate_names', [])),
                    self.safe_json_dumps(record.get('alternate_issns', [])),
                    self.safe_json_dumps(record.get('alternate_urls', [])),
                    record.get('issn'),
                    record.get('url')
                )
                
                # Validate required fields
                if not venue_data[0] or not venue_data[1]:  # id and name required
                    self.error_counts['venue_missing_required'] += 1
                    continue
                
                batch.append(venue_data)
                
                if len(batch) >= self.batch_size:
                    self._insert_venue_batch(cursor, batch)
                    batch = []
                    
            except Exception as e:
                self.error_counts['venue_processing'] += 1
                logger.warning(f"Error processing venue record from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batch
        if batch:
            self._insert_venue_batch(cursor, batch)
        
        self.schema.conn.commit()
        logger.info(f"Processed {self.processed_counts['venues']} venues")
    
    def _insert_venue_batch(self, cursor, batch: List[tuple]):
        """Insert batch of venues"""
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO publication_venues 
                (venue_id, name, type, alternate_names, alternate_issns, alternate_urls, issn, venue_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            self.processed_counts['venues'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting venue batch: {e}")
            self.error_counts['venue_insert'] += len(batch)
    
    def process_authors(self):
        """Process authors data"""
        logger.info("Processing authors...")
        
        cursor = self.schema.conn.cursor()
        batch = []
        
        for record, file_path, line_num in self.read_jsonl_gz_files("authors/part_*.jsonl.gz"):
            try:
                author_data = (
                    record.get('authorid'),
                    record.get('name'),
                    self.safe_json_dumps(record.get('aliases')),
                    self.safe_json_dumps(record.get('affiliations')),
                    record.get('homepage'),
                    record.get('papercount', 0),
                    record.get('citationcount', 0),
                    record.get('hindex', 0),
                    self.safe_json_dumps(record.get('externalids')),
                    record.get('url')
                )
                
                # Validate required fields
                if not author_data[0] or not author_data[1]:  # authorid and name required
                    self.error_counts['author_missing_required'] += 1
                    continue
                
                batch.append(author_data)
                
                if len(batch) >= self.batch_size:
                    self._insert_author_batch(cursor, batch)
                    batch = []
                    
            except Exception as e:
                self.error_counts['author_processing'] += 1
                logger.warning(f"Error processing author record from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batch
        if batch:
            self._insert_author_batch(cursor, batch)
        
        self.schema.conn.commit()
        logger.info(f"Processed {self.processed_counts['authors']} authors")
    
    def _insert_author_batch(self, cursor, batch: List[tuple]):
        """Insert batch of authors"""
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO authors 
                (author_id, name, aliases, affiliations, homepage, paper_count, 
                 citation_count, h_index, external_ids, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            self.processed_counts['authors'] += len(batch)
        except Exception as e:
            logger.error(f"Error inserting author batch: {e}")
            self.error_counts['author_insert'] += len(batch)
    
    def process_papers(self):
        """Process papers data"""
        logger.info("Processing papers...")
        
        cursor = self.schema.conn.cursor()
        paper_batch = []
        author_batch = []
        
        for record, file_path, line_num in self.read_jsonl_gz_files("papers/part_*.jsonl.gz"):
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
                    None,  # fields_of_study - not in current data
                    self.safe_json_dumps(record.get('s2fieldsofstudy')),
                    self.safe_json_dumps(record.get('publicationtypes')),
                    self.safe_json_dumps(record.get('externalids')),
                    self.safe_json_dumps(record.get('journal')),
                    record.get('url')
                )
                
                # Validate required fields
                if not paper_data[2]:  # title required
                    self.error_counts['paper_missing_title'] += 1
                    continue
                
                paper_batch.append(paper_data)
                
                # Process authors for this paper
                authors = record.get('authors', [])
                for pos, author in enumerate(authors, 1):
                    if isinstance(author, dict) and author.get('authorId'):
                        author_rel_data = (
                            paper_id,
                            author.get('authorId'),
                            pos,
                            False  # is_corresponding_author - not in current data
                        )
                        author_batch.append(author_rel_data)
                
                # Insert batches when they reach size limit
                if len(paper_batch) >= self.batch_size:
                    self._insert_paper_batch(cursor, paper_batch)
                    paper_batch = []
                
                if len(author_batch) >= self.batch_size:
                    self._insert_paper_author_batch(cursor, author_batch)
                    author_batch = []
                    
            except Exception as e:
                self.error_counts['paper_processing'] += 1
                logger.warning(f"Error processing paper record from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batches
        if paper_batch:
            self._insert_paper_batch(cursor, paper_batch)
        if author_batch:
            self._insert_paper_author_batch(cursor, author_batch)
        
        self.schema.conn.commit()
        logger.info(f"Processed {self.processed_counts['papers']} papers and {self.processed_counts['paper_authors']} paper-author relationships")
    
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
    
    def process_citations(self):
        """Process citations data"""
        logger.info("Processing citations...")
        
        cursor = self.schema.conn.cursor()
        batch = []
        
        for record, file_path, line_num in self.read_jsonl_gz_files("citations/part_*.jsonl.gz"):
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
                
                if len(batch) >= self.batch_size:
                    self._insert_citation_batch(cursor, batch)
                    batch = []
                    
            except Exception as e:
                self.error_counts['citation_processing'] += 1
                logger.warning(f"Error processing citation record from {file_path}:{line_num}: {e}")
                continue
        
        # Insert remaining batch
        if batch:
            self._insert_citation_batch(cursor, batch)
        
        self.schema.conn.commit()
        logger.info(f"Processed {self.processed_counts['citations']} citations")
    
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
    
    def process_all_data(self):
        """Process all data types in optimal order"""
        self.start_time = time.time()
        
        try:
            # Connect to database
            self.schema.connect()
            
            # Create schema if it doesn't exist
            logger.info("Creating database schema...")
            self.schema.create_schema()
            
            # Process in dependency order
            self.process_publication_venues()
            self.process_authors()
            self.process_papers()
            self.process_citations()
            
            # Create indexes for performance
            logger.info("Creating database indexes...")
            self.schema.create_indexes()
            
            # Optimize database
            logger.info("Optimizing database...")
            self.schema.vacuum_and_analyze()
            
            # Print final statistics
            self._print_final_stats()
            
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.schema.disconnect()
    
    def _print_final_stats(self):
        """Print final processing statistics"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Get final database stats
        stats = self.schema.get_table_stats()
        logger.info("\nDatabase Statistics:")
        for table, count in stats.items():
            logger.info(f"  {table}: {count:,} rows")
        
        # Print processed counts
        logger.info("\nProcessed Records:")
        for data_type, count in self.processed_counts.items():
            logger.info(f"  {data_type}: {count:,}")
        
        # Print error counts
        if self.error_counts:
            logger.info("\nErrors Encountered:")
            for error_type, count in self.error_counts.items():
                logger.info(f"  {error_type}: {count:,}")
        
        # Calculate processing rates
        total_processed = sum(self.processed_counts.values())
        if total_time > 0:
            rate = total_processed / total_time
            logger.info(f"\nProcessing Rate: {rate:.0f} records/second")

def main():
    """Main processing function"""
    processor = DataProcessor(db_path="s2ag_database.db", batch_size=1000)
    
    try:
        processor.process_all_data()
        logger.info("Database building completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()