#!/usr/bin/env python3
"""
Data validation and integrity checking for S2AG database
Performs comprehensive checks to ensure data quality and consistency
"""

import sqlite3
import json
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time

from database_schema import DatabaseSchema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and integrity checking"""
    
    def __init__(self, db_path: str = "s2ag_database.db"):
        self.db_path = db_path
        self.schema = DatabaseSchema(db_path)
        self.issues = defaultdict(list)
        self.stats = {}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all validation checks and return results"""
        logger.info("Starting comprehensive data validation...")
        start_time = time.time()
        
        try:
            self.schema.connect()
            
            # Basic statistics
            self.collect_basic_stats()
            
            # Data integrity checks
            self.check_referential_integrity()
            self.check_data_consistency()
            self.check_data_quality()
            self.check_json_fields()
            
            # Performance checks
            self.check_index_usage()
            
            # Generate report
            report = self.generate_report()
            
            end_time = time.time()
            logger.info(f"Validation completed in {end_time - start_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
        finally:
            self.schema.disconnect()
    
    def collect_basic_stats(self):
        """Collect basic database statistics"""
        logger.info("Collecting basic statistics...")
        
        cursor = self.schema.conn.cursor()
        
        # Table sizes
        self.stats['table_sizes'] = self.schema.get_table_stats()
        
        # Null value analysis
        self.stats['null_analysis'] = {}
        
        tables_columns = {
            'authors': ['name', 'aliases', 'affiliations', 'homepage', 'external_ids'],
            'papers': ['title', 'abstract', 'venue', 'publication_venue_id', 'publication_date'],
            'citations': ['cited_paper_id', 'contexts', 'intents'],
            'publication_venues': ['type', 'venue_url']
        }
        
        for table, columns in tables_columns.items():
            self.stats['null_analysis'][table] = {}
            for column in columns:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
                null_count = cursor.fetchone()[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total_count = cursor.fetchone()[0]
                
                null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
                self.stats['null_analysis'][table][column] = {
                    'null_count': null_count,
                    'total_count': total_count,
                    'null_percentage': null_percentage
                }
        
        # Data distribution analysis
        self.stats['distributions'] = {}
        
        # Year distribution in papers
        cursor.execute("""
            SELECT year, COUNT(*) as count 
            FROM papers 
            WHERE year IS NOT NULL 
            GROUP BY year 
            ORDER BY year
        """)
        year_dist = cursor.fetchall()
        self.stats['distributions']['papers_by_year'] = year_dist
        
        # Citation count distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN citation_count = 0 THEN '0'
                    WHEN citation_count BETWEEN 1 AND 10 THEN '1-10'
                    WHEN citation_count BETWEEN 11 AND 100 THEN '11-100'
                    WHEN citation_count BETWEEN 101 AND 1000 THEN '101-1000'
                    ELSE '1000+'
                END as range,
                COUNT(*) as count
            FROM papers
            GROUP BY range
            ORDER BY 
                CASE range
                    WHEN '0' THEN 1
                    WHEN '1-10' THEN 2
                    WHEN '11-100' THEN 3
                    WHEN '101-1000' THEN 4
                    ELSE 5
                END
        """)
        citation_dist = cursor.fetchall()
        self.stats['distributions']['citation_count_ranges'] = citation_dist
    
    def check_referential_integrity(self):
        """Check foreign key constraints and referential integrity"""
        logger.info("Checking referential integrity...")
        
        cursor = self.schema.conn.cursor()
        
        # Check paper-author relationships
        cursor.execute("""
            SELECT COUNT(*) FROM paper_authors pa
            LEFT JOIN papers p ON pa.paper_id = p.paper_id
            WHERE p.paper_id IS NULL
        """)
        orphaned_paper_refs = cursor.fetchone()[0]
        if orphaned_paper_refs > 0:
            self.issues['referential_integrity'].append(
                f"Found {orphaned_paper_refs} paper-author relationships referencing non-existent papers"
            )
        
        cursor.execute("""
            SELECT COUNT(*) FROM paper_authors pa
            LEFT JOIN authors a ON pa.author_id = a.author_id
            WHERE a.author_id IS NULL
        """)
        orphaned_author_refs = cursor.fetchone()[0]
        if orphaned_author_refs > 0:
            self.issues['referential_integrity'].append(
                f"Found {orphaned_author_refs} paper-author relationships referencing non-existent authors"
            )
        
        # Check citation relationships
        cursor.execute("""
            SELECT COUNT(*) FROM citations c
            LEFT JOIN papers p ON c.citing_paper_id = p.paper_id
            WHERE p.paper_id IS NULL
        """)
        orphaned_citing_refs = cursor.fetchone()[0]
        if orphaned_citing_refs > 0:
            self.issues['referential_integrity'].append(
                f"Found {orphaned_citing_refs} citations with non-existent citing papers"
            )
        
        cursor.execute("""
            SELECT COUNT(*) FROM citations c
            LEFT JOIN papers p ON c.cited_paper_id = p.paper_id
            WHERE c.cited_paper_id IS NOT NULL AND p.paper_id IS NULL
        """)
        orphaned_cited_refs = cursor.fetchone()[0]
        if orphaned_cited_refs > 0:
            self.issues['referential_integrity'].append(
                f"Found {orphaned_cited_refs} citations with non-existent cited papers"
            )
        
        # Check publication venue references
        cursor.execute("""
            SELECT COUNT(*) FROM papers p
            LEFT JOIN publication_venues pv ON p.publication_venue_id = pv.venue_id
            WHERE p.publication_venue_id IS NOT NULL AND pv.venue_id IS NULL
        """)
        orphaned_venue_refs = cursor.fetchone()[0]
        if orphaned_venue_refs > 0:
            self.issues['referential_integrity'].append(
                f"Found {orphaned_venue_refs} papers referencing non-existent publication venues"
            )
    
    def check_data_consistency(self):
        """Check for data consistency issues"""
        logger.info("Checking data consistency...")
        
        cursor = self.schema.conn.cursor()
        
        # Check author paper counts vs actual papers
        cursor.execute("""
            SELECT a.author_id, a.name, a.paper_count, COUNT(pa.paper_id) as actual_count
            FROM authors a
            LEFT JOIN paper_authors pa ON a.author_id = pa.author_id
            GROUP BY a.author_id, a.name, a.paper_count
            HAVING a.paper_count != COUNT(pa.paper_id)
            LIMIT 10
        """)
        inconsistent_paper_counts = cursor.fetchall()
        if inconsistent_paper_counts:
            self.issues['data_consistency'].append(
                f"Found {len(inconsistent_paper_counts)} authors with inconsistent paper counts"
            )
            for author_id, name, stored_count, actual_count in inconsistent_paper_counts[:5]:
                self.issues['data_consistency'].append(
                    f"  Author {author_id} ({name}): stored={stored_count}, actual={actual_count}"
                )
        
        # Check paper citation counts vs actual citations
        cursor.execute("""
            SELECT p.paper_id, p.title, p.citation_count, COUNT(c.citation_id) as actual_count
            FROM papers p
            LEFT JOIN citations c ON p.paper_id = c.cited_paper_id
            GROUP BY p.paper_id, p.title, p.citation_count
            HAVING p.citation_count != COUNT(c.citation_id)
            LIMIT 10
        """)
        inconsistent_citation_counts = cursor.fetchall()
        if inconsistent_citation_counts:
            self.issues['data_consistency'].append(
                f"Found {len(inconsistent_citation_counts)} papers with inconsistent citation counts"
            )
            for paper_id, title, stored_count, actual_count in inconsistent_citation_counts[:5]:
                self.issues['data_consistency'].append(
                    f"  Paper {paper_id} ({title[:50]}...): stored={stored_count}, actual={actual_count}"
                )
        
        # Check for duplicate records
        cursor.execute("SELECT author_id, COUNT(*) FROM authors GROUP BY author_id HAVING COUNT(*) > 1")
        duplicate_authors = cursor.fetchall()
        if duplicate_authors:
            self.issues['data_consistency'].append(
                f"Found {len(duplicate_authors)} duplicate author IDs"
            )
        
        cursor.execute("SELECT paper_id, COUNT(*) FROM papers GROUP BY paper_id HAVING COUNT(*) > 1")
        duplicate_papers = cursor.fetchall()
        if duplicate_papers:
            self.issues['data_consistency'].append(
                f"Found {len(duplicate_papers)} duplicate paper IDs"
            )
    
    def check_data_quality(self):
        """Check for data quality issues"""
        logger.info("Checking data quality...")
        
        cursor = self.schema.conn.cursor()
        
        # Check for empty or suspicious titles
        cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE title IS NULL OR title = '' OR LENGTH(title) < 5
        """)
        bad_titles = cursor.fetchone()[0]
        if bad_titles > 0:
            self.issues['data_quality'].append(
                f"Found {bad_titles} papers with missing or suspiciously short titles"
            )
        
        # Check for empty author names
        cursor.execute("""
            SELECT COUNT(*) FROM authors 
            WHERE name IS NULL OR name = '' OR LENGTH(name) < 2
        """)
        bad_names = cursor.fetchone()[0]
        if bad_names > 0:
            self.issues['data_quality'].append(
                f"Found {bad_names} authors with missing or suspiciously short names"
            )
        
        # Check for unrealistic years
        cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE year IS NOT NULL AND (year < 1900 OR year > 2030)
        """)
        bad_years = cursor.fetchone()[0]
        if bad_years > 0:
            self.issues['data_quality'].append(
                f"Found {bad_years} papers with unrealistic publication years"
            )
        
        # Check for negative counts
        cursor.execute("""
            SELECT COUNT(*) FROM papers 
            WHERE citation_count < 0 OR reference_count < 0 OR influential_citation_count < 0
        """)
        negative_counts = cursor.fetchone()[0]
        if negative_counts > 0:
            self.issues['data_quality'].append(
                f"Found {negative_counts} papers with negative count values"
            )
        
        cursor.execute("""
            SELECT COUNT(*) FROM authors 
            WHERE paper_count < 0 OR citation_count < 0 OR h_index < 0
        """)
        negative_author_counts = cursor.fetchone()[0]
        if negative_author_counts > 0:
            self.issues['data_quality'].append(
                f"Found {negative_author_counts} authors with negative count values"
            )
    
    def check_json_fields(self):
        """Check JSON field validity"""
        logger.info("Checking JSON field validity...")
        
        cursor = self.schema.conn.cursor()
        json_fields = [
            ('authors', 'aliases'),
            ('authors', 'affiliations'),
            ('authors', 'external_ids'),
            ('papers', 's2_fields_of_study'),
            ('papers', 'publication_types'),
            ('papers', 'external_ids'),
            ('papers', 'journal_info'),
            ('citations', 'contexts'),
            ('citations', 'intents'),
            ('publication_venues', 'alternate_names'),
            ('publication_venues', 'alternate_issns'),
            ('publication_venues', 'alternate_urls')
        ]
        
        for table, column in json_fields:
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table} 
                WHERE {column} IS NOT NULL AND {column} != 'null'
            """)
            total_non_null = cursor.fetchone()[0]
            
            if total_non_null > 0:
                # Sample a few records to validate JSON
                cursor.execute(f"""
                    SELECT {column} FROM {table} 
                    WHERE {column} IS NOT NULL AND {column} != 'null'
                    LIMIT 100
                """)
                
                invalid_json_count = 0
                for (json_str,) in cursor.fetchall():
                    try:
                        json.loads(json_str)
                    except (json.JSONDecodeError, TypeError):
                        invalid_json_count += 1
                
                if invalid_json_count > 0:
                    self.issues['data_quality'].append(
                        f"Found {invalid_json_count}/100 invalid JSON values in {table}.{column}"
                    )
    
    def check_index_usage(self):
        """Check if indexes are being used effectively"""
        logger.info("Checking index usage...")
        
        cursor = self.schema.conn.cursor()
        
        # Get index information
        cursor.execute("PRAGMA index_list('authors')")
        author_indexes = cursor.fetchall()
        
        cursor.execute("PRAGMA index_list('papers')")
        paper_indexes = cursor.fetchall()
        
        cursor.execute("PRAGMA index_list('citations')")
        citation_indexes = cursor.fetchall()
        
        self.stats['indexes'] = {
            'authors': len(author_indexes),
            'papers': len(paper_indexes),
            'citations': len(citation_indexes)
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'database_path': self.db_path,
            'statistics': self.stats,
            'issues': dict(self.issues),
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of validation results"""
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        summary = {
            'total_issues_found': total_issues,
            'issues_by_category': {category: len(issues) for category, issues in self.issues.items()},
            'overall_health': 'Good' if total_issues == 0 else 'Issues Found' if total_issues < 10 else 'Needs Attention'
        }
        
        if self.stats.get('table_sizes'):
            summary['total_records'] = sum(self.stats['table_sizes'].values())
        
        return summary
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "=" * 80)
        print("DATABASE VALIDATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Database: {report['database_path']}")
        
        # Summary
        summary = report['summary']
        print(f"\nOVERALL HEALTH: {summary['overall_health']}")
        print(f"Total Issues Found: {summary['total_issues_found']}")
        
        if summary.get('total_records'):
            print(f"Total Records: {summary['total_records']:,}")
        
        # Table sizes
        if 'table_sizes' in report['statistics']:
            print(f"\nTABLE SIZES:")
            for table, size in report['statistics']['table_sizes'].items():
                print(f"  {table}: {size:,} rows")
        
        # Issues by category
        if report['issues']:
            print(f"\nISSUES BY CATEGORY:")
            for category, issues in report['issues'].items():
                print(f"\n{category.upper().replace('_', ' ')} ({len(issues)} issues):")
                for issue in issues:
                    print(f"  - {issue}")
        
        # Data quality highlights
        if 'null_analysis' in report['statistics']:
            print(f"\nNULL VALUE ANALYSIS:")
            for table, columns in report['statistics']['null_analysis'].items():
                print(f"\n{table}:")
                for column, stats in columns.items():
                    if stats['null_percentage'] > 50:
                        print(f"  {column}: {stats['null_percentage']:.1f}% null ({stats['null_count']:,}/{stats['total_count']:,})")
        
        print("\n" + "=" * 80)

def main():
    """Run validation checks"""
    validator = DataValidator("s2ag_database.db")
    
    try:
        report = validator.run_all_checks()
        validator.print_report(report)
        
        # Save report to file
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Validation report saved to validation_report.json")
        
        return report['summary']['overall_health'] == 'Good'
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)