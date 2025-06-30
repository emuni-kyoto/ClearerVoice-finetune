#!/usr/bin/env python3
"""Test script to verify GCS access and list files."""

import logging
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gcs_access():
    """Test GCS access and list some files."""
    bucket_name = "audio_datasets_emuni"
    prefix = "podcast/parquet_processed/ja"
    
    try:
        logger.info(f"Testing access to gs://{bucket_name}/{prefix}")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List first 5 parquet files
        logger.info("\nListing first 5 parquet files:")
        count = 0
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith('.parquet'):
                logger.info(f"  - {blob.name}")
                count += 1
                if count >= 5:
                    break
        
        if count == 0:
            logger.warning("No parquet files found!")
        
        # Test a specific file path
        test_path = "podcast/audio_files_processed/ja/571eadf2-7823-4ce6-a08b-e06339203367/2a8577c8_4740_4fcf_9504_956f1cd73188.wav"
        logger.info(f"\nChecking if specific file exists: {test_path}")
        
        blob = bucket.blob(test_path)
        if blob.exists():
            logger.info(f"✓ File exists! Size: {blob.size / 1024 / 1024:.2f} MB")
        else:
            logger.error("✗ File does not exist")
            
            # Try to list files in the directory
            dir_prefix = "podcast/audio_files_processed/ja/571eadf2-7823-4ce6-a08b-e06339203367/"
            logger.info(f"\nListing files in directory: {dir_prefix}")
            dir_count = 0
            for blob in bucket.list_blobs(prefix=dir_prefix, max_results=5):
                logger.info(f"  - {blob.name}")
                dir_count += 1
            
            if dir_count == 0:
                logger.warning("No files found in this directory")
        
    except Exception as e:
        logger.error(f"Failed to access GCS: {e}")
        logger.error("Make sure you have authenticated with: gcloud auth application-default login")

if __name__ == "__main__":
    test_gcs_access()