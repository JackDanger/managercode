#!/usr/bin/env python3
"""
Production script to export Slack Q&A data for fine-tuning.

Usage:
    python export_training_data.py [--max-pairs-per-channel N] [--output-dir DIR]
"""

import argparse
import os
import sys
from datetime import datetime

# Add the parent directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import DatabaseManager
from exporter import Exporter

def main():
    parser = argparse.ArgumentParser(description="Export Slack Q&A data for LLM fine-tuning")
    parser.add_argument(
        "--max-pairs-per-channel", 
        type=int, 
        default=100,
        help="Maximum Q&A pairs to extract per channel (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"fine_tune_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for the JSONL file"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="db.sqlite3",
        help="Path to the Slack archive database"
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"Error: Database not found at {args.db_path}")
        print("Please run the importer first to create the database.")
        sys.exit(1)
    
    # Connect to database
    print(f"Connecting to database: {args.db_path}")
    db = DatabaseManager(args.db_path)
    db.connect()
    
    # Create exporter
    exporter = Exporter(db)
    
    # Run export
    print(f"Output directory: {args.output_dir}")
    print(f"Max pairs per channel: {args.max_pairs_per_channel}")
    
    try:
        exporter.export_fine_tuning_data(args.output_dir, args.max_pairs_per_channel)
        
        # Show the results
        output_file = os.path.join(args.output_dir, "slack_qa_training.jsonl")
        metadata_file = os.path.join(args.output_dir, "export_metadata.json")
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            with open(output_file, 'r') as f:
                line_count = sum(1 for _ in f)
            
            print(f"\n✓ Export successful!")
            print(f"  Training data: {output_file}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Training examples: {line_count:,}")
            print(f"  Metadata: {metadata_file}")
            
            print("\nNext steps:")
            print("1. Review the export_metadata.json for statistics")
            print("2. Fine-tune your model with:")
            print(f"\n   python ../fine-tune-anything/main.py --train \\")
            print(f"     --jsonl-file {os.path.abspath(output_file)} \\")
            print(f"     --output-dir ./finetuned_slack_model \\")
            print(f"     --max-length 1024 \\")
            print(f"     --lora-r 32 \\")
            print(f"     --epochs 2")
        else:
            print("✗ Export failed: No output file created")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Export failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close database connection
        db.close()

if __name__ == "__main__":
    main() 