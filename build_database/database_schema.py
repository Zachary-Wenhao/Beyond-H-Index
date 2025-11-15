#!/usr/bin/env python3
"""
Database schema definition and initialization for S2AG data
Robust schema designed for academic paper analysis with proper indexing
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSchema:
    """Handles database schema creation and management"""
    
    def __init__(self, db_path: str = "s2ag_database.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to database with optimized settings"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")  # Better performance for concurrent access
        self.conn.execute("PRAGMA synchronous = NORMAL")  # Balanced performance/safety
        self.conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        self.conn.execute("PRAGMA temp_store = MEMORY")
        return self.conn
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_schema(self):
        """Create all tables with proper schema"""
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Drop existing tables if they exist (for clean rebuild)
        tables_to_drop = [
            'citations', 'paper_authors', 'papers', 'authors', 'publication_venues'
        ]
        
        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Create Authors table
        cursor.execute("""
            CREATE TABLE authors (
                author_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                aliases TEXT,  -- JSON array
                affiliations TEXT,  -- JSON array 
                homepage TEXT,
                paper_count INTEGER DEFAULT 0,
                citation_count INTEGER DEFAULT 0,
                h_index INTEGER DEFAULT 0,
                external_ids TEXT,  -- JSON object
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Publication Venues table
        cursor.execute("""
            CREATE TABLE publication_venues (
                venue_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                alternate_names TEXT,  -- JSON array
                alternate_issns TEXT,  -- JSON array
                alternate_urls TEXT,   -- JSON array
                issn TEXT,
                venue_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Papers table
        cursor.execute("""
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,  -- Using corpusid as string
                corpus_id INTEGER UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,  -- Not in current data but keeping for future
                year INTEGER,
                publication_date TEXT,
                venue TEXT,
                publication_venue_id TEXT,
                reference_count INTEGER DEFAULT 0,
                citation_count INTEGER DEFAULT 0,
                influential_citation_count INTEGER DEFAULT 0,
                is_open_access BOOLEAN DEFAULT FALSE,
                fields_of_study TEXT,  -- JSON array
                s2_fields_of_study TEXT,  -- JSON array  
                publication_types TEXT,  -- JSON array
                external_ids TEXT,  -- JSON object
                journal_info TEXT,  -- JSON object
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (publication_venue_id) REFERENCES publication_venues(venue_id)
            )
        """)
        
        # Create Paper-Authors relationship table (many-to-many with position)
        cursor.execute("""
            CREATE TABLE paper_authors (
                paper_id TEXT,
                author_id TEXT,
                author_position INTEGER,  -- 1st, 2nd, etc.
                is_corresponding_author BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (paper_id, author_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
                FOREIGN KEY (author_id) REFERENCES authors(author_id)
            )
        """)
        
        # Create Citations table (paper-to-paper relationships)
        cursor.execute("""
            CREATE TABLE citations (
                citation_id INTEGER PRIMARY KEY,
                citing_paper_id TEXT,  -- corpus_id as string
                cited_paper_id TEXT,   -- corpus_id as string (can be NULL)
                contexts TEXT,  -- JSON array
                intents TEXT,   -- JSON array
                is_influential BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (citing_paper_id) REFERENCES papers(paper_id),
                FOREIGN KEY (cited_paper_id) REFERENCES papers(paper_id)
            )
        """)
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
    def create_indexes(self):
        """Create optimized indexes for common queries"""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        
        # Author indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_authors_citation_count ON authors(citation_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_authors_h_index ON authors(h_index DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_authors_paper_count ON authors(paper_count DESC)")
        
        # Paper indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_corpus_id ON papers(corpus_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_citation_count ON papers(citation_count DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_venue_id ON papers(publication_venue_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
        
        # Paper-Authors indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_authors_paper ON paper_authors(paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_authors_position ON paper_authors(author_position)")
        
        # Citation indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_influential ON citations(is_influential)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_citations_pair ON citations(citing_paper_id, cited_paper_id)")
        
        # Venue indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_venues_name ON publication_venues(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_venues_type ON publication_venues(type)")
        
        self.conn.commit()
        logger.info("Database indexes created successfully")
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        stats = {}
        
        tables = ['authors', 'papers', 'publication_venues', 'paper_authors', 'citations']
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count
            except Exception as e:
                logger.error(f"Error getting stats for {table}: {e}")
                stats[table] = -1
        
        return stats
    
    def vacuum_and_analyze(self):
        """Optimize database after bulk inserts"""
        if not self.conn:
            self.connect()
            
        logger.info("Running VACUUM and ANALYZE...")
        self.conn.execute("VACUUM")
        self.conn.execute("ANALYZE")
        self.conn.commit()
        logger.info("Database optimization complete")

def main():
    """Test schema creation"""
    schema = DatabaseSchema("test_s2ag.db")
    
    try:
        schema.connect()
        schema.create_schema()
        schema.create_indexes()
        
        stats = schema.get_table_stats()
        print("Database created successfully!")
        print("Table statistics:")
        for table, count in stats.items():
            print(f"  {table}: {count} rows")
            
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise
    finally:
        schema.disconnect()

if __name__ == "__main__":
    main()