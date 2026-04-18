"""
Data Drift Detection & Quality Monitoring
Demonstrates data quality checks and drift detection without needing Airflow running
Can be integrated into Airflow DAGs later
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

class DataDriftMonitor:
    """Monitor data quality and detect drift in product data"""
    
    def __init__(self, baseline_data_path: str):
        """Initialize with baseline/reference data"""
        self.baseline = pd.read_csv(baseline_data_path)
        self.baseline_stats = self._calculate_statistics(self.baseline)
        
    def _calculate_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate statistical properties of dataset"""
        stats = {
            'count': len(df),
            'name_length_mean': df['Name'].str.len().mean(),
            'name_length_std': df['Name'].str.len().std(),
            'desc_length_mean': df['Description'].str.len().mean(),
            'desc_length_std': df['Description'].str.len().std(),
            'category_distribution': df['Category'].value_counts().to_dict(),
            'unique_names': df['Name'].nunique(),
            'timestamp': datetime.now().isoformat()
        }
        return stats
    
    def detect_drift(self, new_data_path: str, threshold: float = 0.3) -> dict:
        """
        Detect drift between baseline and new data
        
        Args:
            new_data_path: Path to new dataset
            threshold: Drift alert threshold (0-1)
            
        Returns:
            dict with drift metrics and alerts
        """
        new_data = pd.read_csv(new_data_path)
        new_stats = self._calculate_statistics(new_data)
        
        # Calculate drift scores
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_count': self.baseline_stats['count'],
            'new_count': new_stats['count'],
            'drifts': [],
            'alerts': []
        }
        
        # Check sample size drift
        size_drift = abs(new_stats['count'] - self.baseline_stats['count']) / self.baseline_stats['count']
        if size_drift > threshold:
            drift_report['alerts'].append({
                'type': 'SIZE_DRIFT',
                'severity': 'HIGH',
                'message': f"Sample size changed by {size_drift*100:.1f}%"
            })
        
        # Check name length drift
        name_drift = abs(new_stats['name_length_mean'] - self.baseline_stats['name_length_mean']) / self.baseline_stats['name_length_mean']
        drift_report['drifts'].append({
            'feature': 'name_length',
            'baseline_mean': self.baseline_stats['name_length_mean'],
            'new_mean': new_stats['name_length_mean'],
            'drift_score': name_drift
        })
        
        if name_drift > threshold:
            drift_report['alerts'].append({
                'type': 'NAME_LENGTH_DRIFT',
                'severity': 'MEDIUM',
                'message': f"Product name length drifted by {name_drift*100:.1f}%"
            })
        
        # Check description length drift
        desc_drift = abs(new_stats['desc_length_mean'] - self.baseline_stats['desc_length_mean']) / self.baseline_stats['desc_length_mean']
        drift_report['drifts'].append({
            'feature': 'description_length',
            'baseline_mean': self.baseline_stats['desc_length_mean'],
            'new_mean': new_stats['desc_length_mean'],
            'drift_score': desc_drift
        })
        
        if desc_drift > threshold:
            drift_report['alerts'].append({
                'type': 'DESCRIPTION_LENGTH_DRIFT',
                'severity': 'MEDIUM',
                'message': f"Description length drifted by {desc_drift*100:.1f}%"
            })
        
        # Check category distribution drift (Kullback-Leibler divergence simplified)
        baseline_cats = set(self.baseline_stats['category_distribution'].keys())
        new_cats = set(new_stats['category_distribution'].keys())
        
        if baseline_cats != new_cats:
            drift_report['alerts'].append({
                'type': 'CATEGORY_DRIFT',
                'severity': 'HIGH',
                'message': f"New categories detected: {new_cats - baseline_cats}, Missing: {baseline_cats - new_cats}"
            })
        
        # Overall drift score
        drift_report['overall_drift_score'] = (name_drift + desc_drift) / 2
        drift_report['status'] = 'DRIFT_DETECTED' if drift_report['alerts'] else 'NO_DRIFT'
        
        return drift_report
    
    def check_data_quality(self, data_path: str) -> dict:
        """
        Check data quality issues
        
        Returns:
            dict with quality metrics
        """
        df = pd.read_csv(data_path)
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'issues': [],
            'quality_score': 1.0
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            quality_report['issues'].append({
                'type': 'MISSING_VALUES',
                'details': missing[missing > 0].to_dict()
            })
            quality_report['quality_score'] -= 0.2
        
        # Check for duplicate names
        duplicates = df['Name'].duplicated().sum()
        if duplicates > 0:
            quality_report['issues'].append({
                'type': 'DUPLICATES',
                'count': int(duplicates)
            })
            quality_report['quality_score'] -= 0.1
        
        # Check for empty/short descriptions
        short_desc = (df['Description'].str.len() < 10).sum()
        if short_desc > 0:
            quality_report['issues'].append({
                'type': 'SHORT_DESCRIPTIONS',
                'count': int(short_desc)
            })
            quality_report['quality_score'] -= 0.1
        
        # Check for unusual characters
        unusual = df['Name'].str.contains('[^a-zA-Z0-9 \-]', regex=True).sum()
        if unusual > 0:
            quality_report['issues'].append({
                'type': 'UNUSUAL_CHARACTERS',
                'count': int(unusual)
            })
            quality_report['quality_score'] -= 0.05
        
        quality_report['status'] = 'PASS' if quality_report['quality_score'] >= 0.8 else 'FAIL'
        
        return quality_report


def generate_demo_data():
    """Generate sample data to demonstrate drift detection"""
    print("📊 Generating demo datasets...")
    
    # Original dataset (already exists)
    baseline_path = "data/products_sample.csv"
    
    # Create drifted dataset
    df = pd.read_csv(baseline_path)
    
    # Simulate drift: longer names and descriptions
    drifted = df.copy()
    drifted['Name'] = drifted['Name'] + " - Premium Edition"
    drifted['Description'] = drifted['Description'] + " With advanced features and enhanced performance."
    
    drifted_path = "data/products_drifted.csv"
    drifted.to_csv(drifted_path, index=False)
    print(f"✅ Created drifted dataset: {drifted_path}")
    
    return baseline_path, drifted_path


def run_drift_detection():
    """Main function to demonstrate drift detection"""
    print("="*70)
    print("🔍 DATA DRIFT DETECTION & QUALITY MONITORING")
    print("="*70)
    
    #Generate demo data
    baseline_path, drifted_path = generate_demo_data()
    
    # Initialize monitor
    print(f"\n📈 Initializing monitor with baseline: {baseline_path}")
    monitor = DataDriftMonitor(baseline_path)
    
    # Check quality of baseline
    print("\n🔎 Checking baseline data quality...")
    quality_report = monitor.check_data_quality(baseline_path)
    print(f"   Status: {quality_report['status']}")
    print(f"   Quality Score: {quality_report['quality_score']:.2f}")
    print(f"   Issues Found: {len(quality_report['issues'])}")
    
    if quality_report['issues']:
        for issue in quality_report['issues']:
            print(f"      - {issue['type']}: {issue.get('count', 'See details')}")
    
    # Detect drift
    print(f"\n🎯 Detecting drift in new data: {drifted_path}")
    drift_report = monitor.detect_drift(drifted_path, threshold=0.1)
    
    print(f"\n📊 DRIFT REPORT")
    print(f"   Overall Status: {drift_report['status']}")
    print(f"   Overall Drift Score: {drift_report['overall_drift_score']:.4f}")
    print(f"   Baseline Samples: {drift_report['baseline_count']}")
    print(f"   New Samples: {drift_report['new_count']}")
    
    print(f"\n   Drift Metrics:")
    for drift in drift_report['drifts']:
        print(f"      • {drift['feature']}: {drift['drift_score']:.4f}")
        print(f"         Baseline: {drift['baseline_mean']:.2f} → New: {drift['new_mean']:.2f}")
    
    if drift_report['alerts']:
        print(f"\n   ⚠️  ALERTS ({len(drift_report['alerts'])} detected):")
        for alert in drift_report['alerts']:
            print(f"      🔔 [{alert['severity']}] {alert['type']}")
            print(f"         {alert['message']}")
    else:
        print(f"\n   ✅ No drift alerts")
    
    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    quality_file = reports_dir / "data_quality_report.json"
    with open(quality_file, 'w') as f:
        json.dump(quality_report, f, indent=2)
    print(f"\n💾 Quality report saved: {quality_file}")
    
    drift_file = reports_dir / "drift_detection_report.json"
    with open(drift_file, 'w') as f:
        json.dump(drift_report, f, indent=2)
    print(f"💾 Drift report saved: {drift_file}")
    
    print("\n" + "="*70)
    print("✅ DATA MONITORING COMPLETE")
    print("="*70)
    
    return quality_report, drift_report


if __name__ == "__main__":
    quality, drift = run_drift_detection()
