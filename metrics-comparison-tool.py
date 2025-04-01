import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import argparse
import json

class ResearcherMetricsComparator:
    """
    A tool to compare and rank researchers based on different metrics.
    """
    def __init__(self, data_path):
        """
        Initialize the comparator with researcher metrics data.
        
        Args:
            data_path: Path to the CSV file containing researcher metrics
        """
        self.data = pd.read_csv(data_path)
        self.researcher_papers = {}
        self.load_paper_data()
        print(f"Loaded data for {len(self.data)} researchers")
    
    def load_paper_data(self):
        """Load detailed paper data for each researcher from JSON files."""
        paper_data_dir = "researcher_papers"
        if not os.path.exists(paper_data_dir):
            print(f"Paper data directory not found: {paper_data_dir}")
            return
        
        for filename in os.listdir(paper_data_dir):
            if filename.endswith(".json"):
                author_id = filename.replace(".json", "")
                with open(os.path.join(paper_data_dir, filename), 'r') as f:
                    self.researcher_papers[author_id] = json.load(f)
        
        print(f"Loaded detailed paper data for {len(self.researcher_papers)} researchers")
    
    def get_available_metrics(self):
        """Get list of available metrics to rank researchers by."""
        # Exclude columns that aren't metrics
        exclude_cols = ['author_id', 'name', 'institution', 'papers', 'first_publication_year', 
                        'last_publication_year']
        
        # Include all numeric columns except those with 'rank', 'diff', or 'norm' in their name
        metrics = [col for col in self.data.columns 
                  if col not in exclude_cols 
                  and not any(x in col for x in ['_rank', 'diff_', '_norm'])
                  and pd.api.types.is_numeric_dtype(self.data[col])]
        
        return metrics
    
    def rank_by_metric(self, metric_name, n=None, ascending=False):
        """
        Rank researchers by a specific metric.
        
        Args:
            metric_name: Name of the metric to rank by
            n: Number of researchers to return (None for all)
            ascending: Whether to sort in ascending order (default: False, higher is better)
            
        Returns:
            DataFrame of ranked researchers
        """
        if metric_name not in self.data.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
        
        # Sort researchers by the specified metric
        ranked = self.data.sort_values(by=metric_name, ascending=ascending)
        
        # Select columns to display
        display_cols = ['name', metric_name]
        
        # Add institution if available
        if 'institution' in self.data.columns:
            display_cols.append('institution')
        
        # Add h-index for reference if available and not the current metric
        if 'h_index' in self.data.columns and metric_name != 'h_index':
            display_cols.append('h_index')
        
        # Add total papers and citations if available
        for col in ['total_paper_count', 'total_citation_count', 'career_span']:
            if col in self.data.columns and col != metric_name:
                display_cols.append(col)
        
        # Select top n if specified
        if n is not None:
            ranked = ranked.head(n)
        
        return ranked[display_cols]
    
    def compare_metrics(self, metric1, metric2, n=50):
        """
        Compare two metrics by showing their correlation and top researchers.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            n: Number of data points to use (default: 50)
            
        Returns:
            Correlation coefficient and scatter plot
        """
        if metric1 not in self.data.columns or metric2 not in self.data.columns:
            raise ValueError(f"Metrics '{metric1}' or '{metric2}' not found in data")
        
        # Calculate correlation
        correlation = self.data[metric1].corr(self.data[metric2])
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=metric1, y=metric2, data=self.data)
        
        # Add trend line
        sns.regplot(x=metric1, y=metric2, data=self.data, scatter=False, color='red')
        
        # Add correlation annotation
        plt.annotate(f'Correlation: {correlation:.4f}', xy=(0.05, 0.95), 
                    xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Add labels for top n researchers
        top_researchers = self.data.sort_values(by=[metric1, metric2], ascending=False).head(n)
        
        for idx, row in top_researchers.iterrows():
            plt.text(row[metric1], row[metric2], row['name'], 
                    fontsize=8, alpha=0.7, ha='right', va='bottom')
        
        plt.title(f'{metric1} vs {metric2}')
        plt.tight_layout()
        
        return correlation, plt
    
    def get_researcher_details(self, researcher_name):
        """
        Get detailed metrics for a specific researcher.
        
        Args:
            researcher_name: Name of the researcher
            
        Returns:
            Dictionary of researcher details
        """
        # Find the researcher in the data
        researcher = self.data[self.data['name'] == researcher_name]
        
        if len(researcher) == 0:
            raise ValueError(f"Researcher '{researcher_name}' not found in data")
        
        # Get the first match (in case there are duplicates)
        researcher = researcher.iloc[0]
        
        # Get author ID to retrieve paper data
        author_id = researcher.get('author_id')
        
        # Create a dictionary of basic metrics
        details = {
            'name': researcher_name,
            'metrics': {}
        }
        
        # Add all metrics
        for metric in self.get_available_metrics():
            if metric in researcher:
                details['metrics'][metric] = researcher[metric]
        
        # Add institution if available
        if 'institution' in researcher:
            details['institution'] = researcher['institution']
        
        # Add paper details if available
        if author_id in self.researcher_papers:
            details['papers'] = self.researcher_papers[author_id]
        
        return details
    
    def get_rank_for_researcher(self, researcher_name, metric_name):
        """
        Get the rank of a specific researcher for a specific metric.
        
        Args:
            researcher_name: Name of the researcher
            metric_name: Name of the metric
            
        Returns:
            Rank of the researcher for the metric
        """
        if metric_name not in self.data.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
        
        # Create a copy of the data with ranks
        ranked_data = self.data.copy()
        ranked_data[f'{metric_name}_rank'] = ranked_data[metric_name].rank(ascending=False, method='min')
        
        # Find the researcher in the data
        researcher = ranked_data[ranked_data['name'] == researcher_name]
        
        if len(researcher) == 0:
            raise ValueError(f"Researcher '{researcher_name}' not found in data")
        
        # Get the rank
        rank = researcher[f'{metric_name}_rank'].iloc[0]
        
        return int(rank)
    
    def get_researchers_by_rank_range(self, metric_name, min_rank, max_rank):
        """
        Get researchers within a specific rank range for a metric.
        
        Args:
            metric_name: Name of the metric
            min_rank: Minimum rank (inclusive)
            max_rank: Maximum rank (inclusive)
            
        Returns:
            DataFrame of researchers in the rank range
        """
        if metric_name not in self.data.columns:
            raise ValueError(f"Metric '{metric_name}' not found in data")
        
        # Create a copy of the data with ranks
        ranked_data = self.data.copy()
        ranked_data[f'{metric_name}_rank'] = ranked_data[metric_name].rank(ascending=False, method='min')
        
        # Filter by rank range
        range_data = ranked_data[(ranked_data[f'{metric_name}_rank'] >= min_rank) & 
                                (ranked_data[f'{metric_name}_rank'] <= max_rank)]
        
        # Sort by rank
        range_data = range_data.sort_values(by=f'{metric_name}_rank')
        
        # Select columns to display
        display_cols = ['name', metric_name, f'{metric_name}_rank']
        
        # Add institution if available
        if 'institution' in self.data.columns:
            display_cols.append('institution')
        
        # Add h-index for reference if available and not the current metric
        if 'h_index' in self.data.columns and metric_name != 'h_index':
            display_cols.append('h_index')
        
        return range_data[display_cols]
    
    def custom_ranking(self, metric_weights):
        """
        Create a custom ranking based on weighted metrics.
        
        Args:
            metric_weights: Dictionary of metric names and their weights
            
        Returns:
            DataFrame of researchers ranked by the custom score
        """
        # Validate metrics
        for metric in metric_weights:
            if metric not in self.data.columns:
                raise ValueError(f"Metric '{metric}' not found in data")
        
        # Create a copy of the data
        ranked_data = self.data.copy()
        
        # Normalize each metric
        for metric in metric_weights:
            min_val = ranked_data[metric].min()
            max_val = ranked_data[metric].max()
            
            if max_val > min_val:  # Avoid division by zero
                ranked_data[f'{metric}_norm'] = (ranked_data[metric] - min_val) / (max_val - min_val)
            else:
                ranked_data[f'{metric}_norm'] = 0
        
        # Calculate custom score
        ranked_data['custom_score'] = 0
        for metric, weight in metric_weights.items():
            ranked_data['custom_score'] += ranked_data[f'{metric}_norm'] * weight
        
        # Sort by custom score
        ranked_data = ranked_data.sort_values(by='custom_score', ascending=False)
        
        # Select columns to display
        display_cols = ['name', 'custom_score']
        
        # Add institution if available
        if 'institution' in self.data.columns:
            display_cols.append('institution')
        
        # Add the weighted metrics
        for metric in metric_weights:
            display_cols.append(metric)
        
        # Add h-index for reference if available and not already included
        if 'h_index' in self.data.columns and 'h_index' not in metric_weights:
            display_cols.append('h_index')
        
        return ranked_data[display_cols]
    
    def print_researcher_rankings(self, researcher_name):
        """
        Print all rankings for a specific researcher.
        
        Args:
            researcher_name: Name of the researcher
        """
        # Find the researcher in the data
        researcher = self.data[self.data['name'] == researcher_name]
        
        if len(researcher) == 0:
            print(f"Researcher '{researcher_name}' not found in data")
            return
        
        print(f"\nRankings for {researcher_name}:")
        print("-" * 60)
        
        # Get all metrics
        metrics = self.get_available_metrics()
        
        # Calculate ranks for each metric
        ranks = []
        for metric in metrics:
            rank = self.get_rank_for_researcher(researcher_name, metric)
            value = researcher[metric].iloc[0]
            ranks.append((metric, value, rank))
        
        # Sort by rank (best ranks first)
        ranks.sort(key=lambda x: x[2])
        
        # Print table
        headers = ["Metric", "Value", "Rank", "Percentile"]
        rows = []
        
        for metric, value, rank in ranks:
            percentile = 100 * (1 - (rank / len(self.data)))
            rows.append([metric, value, rank, f"{percentile:.1f}%"])
        
        print(tabulate(rows, headers=headers, floatfmt=".2f"))
        print("\n")

def main():
    """Command-line interface for the comparator."""
    parser = argparse.ArgumentParser(description="Compare NLP researcher metrics")
    parser.add_argument("--data", required=True, help="Path to CSV file with researcher metrics")
    parser.add_argument("--action", required=True, choices=["rank", "compare", "details", "range", "custom"],
                      help="Action to perform")
    parser.add_argument("--metric", help="Metric to rank by")
    parser.add_argument("--metric2", help="Second metric for comparison")
    parser.add_argument("--researcher", help="Researcher name for details")
    parser.add_argument("--min_rank", type=int, help="Minimum rank for range")
    parser.add_argument("--max_rank", type=int, help="Maximum rank for range")
    parser.add_argument("--weights", help="Metric weights for custom ranking (format: metric1=0.5,metric2=0.3,metric3=0.2)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--top", type=int, default=10, help="Number of top researchers to show")
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ResearcherMetricsComparator(args.data)
    
    # Perform requested action
    if args.action == "rank":
        if not args.metric:
            print("Error: --metric is required for 'rank' action")
            return
        
        result = comparator.rank_by_metric(args.metric, args.top)
        
        print(f"\nTop {args.top} researchers by {args.metric}:")
        print(tabulate(result, headers="keys", showindex=False, floatfmt=".2f"))
        
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    elif args.action == "compare":
        if not args.metric or not args.metric2:
            print("Error: --metric and --metric2 are required for 'compare' action")
            return
        
        correlation, plot = comparator.compare_metrics(args.metric, args.metric2, args.top)
        
        print(f"\nCorrelation between {args.metric} and {args.metric2}: {correlation:.4f}")
        
        if args.output:
            plot.savefig(args.output)
            print(f"\nPlot saved to {args.output}")
        else:
            plt.show()
    
    elif args.action == "details":
        if not args.researcher:
            print("Error: --researcher is required for 'details' action")
            return
        
        details = comparator.get_researcher_details(args.researcher)
        
        print(f"\nDetails for {args.researcher}:")
        print("-" * 60)
        
        if 'institution' in details:
            print(f"Institution: {details['institution']}")
        
        print("\nMetrics:")
        for metric, value in details['metrics'].items():
            rank = comparator.get_rank_for_researcher(args.researcher, metric)
            percentile = 100 * (1 - (rank / len(comparator.data)))
            print(f"  {metric}: {value:.2f} (Rank: {rank}, Percentile: {percentile:.1f}%)")
        
        if 'papers' in details:
            print(f"\nTop cited papers ({len(details['papers'])} total):")
            
            if isinstance(details['papers'], list) and len(details['papers']) > 0:
                # Sort papers by citation count
                papers = sorted(details['papers'], key=lambda p: p.get('citationCount', 0), reverse=True)
                
                # Display top 5 papers
                for i, paper in enumerate(papers[:5]):
                    print(f"  {i+1}. {paper.get('title', 'No title')} ({paper.get('year', 'N/A')})")
                    print(f"     Citations: {paper.get('citationCount', 0)}, Venue: {paper.get('venue', 'N/A')}")
        
        # Print all rankings
        comparator.print_researcher_rankings(args.researcher)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(details, f, indent=2)
            print(f"\nDetails saved to {args.output}")
    
    elif args.action == "range":
        if not args.metric or args.min_rank is None or args.max_rank is None:
            print("Error: --metric, --min_rank, and --max_rank are required for 'range' action")
            return
        
        result = comparator.get_researchers_by_rank_range(args.metric, args.min_rank, args.max_rank)
        
        print(f"\nResearchers ranked {args.min_rank}-{args.max_rank} by {args.metric}:")
        print(tabulate(result, headers="keys", showindex=False, floatfmt=".2f"))
        
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    
    elif args.action == "custom":
        if not args.weights:
            print("Error: --weights is required for 'custom' action")
            return
        
        # Parse weights
        try:
            weights = {}
            for item in args.weights.split(','):
                metric, weight = item.split('=')
                weights[metric] = float(weight)
        except:
            print("Error: Invalid format for --weights. Expected: metric1=0.5,metric2=0.3,metric3=0.2")
            return
        
        result = comparator.custom_ranking(weights)
        
        print(f"\nTop {args.top} researchers by custom ranking:")
        print(f"Weights: {weights}")
        print(tabulate(result.head(args.top), headers="keys", showindex=False, floatfmt=".2f"))
        
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

# python metrics-comparison-tool.py --data nlp_researcher_metrics.csv --action rank --metric h_index --top 20
# python metrics-comparison-tool.py --data nlp_researcher_metrics.csv --action compare --metric h_index --metric2 venue_diversity
# python metrics-comparison-tool.py --data nlp_researcher_metrics.csv --action details --researcher "Aston Zhang"
# python metrics-comparison-tool.py --data nlp_researcher_metrics.csv --action range --metric citations_per_paper --min_rank 20 --max_rank 30

# Create a custom ranking (40% venue diversity, 30% h-index, 30% citations per paper)
# python metrics-comparison-tool.py --data nlp_researcher_metrics.csv --action custom --weights venue_diversity=0.4,h_index=0.3,citations_per_paper=0.3