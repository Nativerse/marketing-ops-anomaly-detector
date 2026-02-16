"""
Marketing Operations Anomaly Detector
======================================
Monitors Salesforce and Marketing Cloud campaign performance to detect issues before
stakeholders complain. Uses statistical anomaly detection to catch:

- API connection failures
- Email deliverability drops
- Lead quality degradation
- Form submission issues
- Campaign tracking breaks

Purpose: Portfolio demonstration of proactive Marketing Ops monitoring
Author: Marketing Operations Engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set random seed
random.seed(42)
np.random.seed(42)


# ============================================================================
# CONFIGURATION - Realistic Marketing Ops Thresholds
# ============================================================================

class OpsConfig:
    """
    Thresholds based on what actually breaks in marketing ops.
    These are conservative - designed to catch real issues, not false alarms.
    """
    
    # Email deliverability alerts
    DELIVERABILITY_DROP_THRESHOLD = 15  # % drop from baseline triggers alert
    BOUNCE_RATE_THRESHOLD = 5  # Over 5% bounce = ISP issue
    UNSUBSCRIBE_SPIKE_THRESHOLD = 3.0  # 3x normal unsubscribe rate
    
    # Lead flow alerts
    LEAD_VOLUME_DROP_THRESHOLD = 30  # % drop in daily leads
    FORM_ABANDONMENT_THRESHOLD = 60  # >60% abandonment = technical issue
    DUPLICATE_LEAD_THRESHOLD = 10  # >10% duplicates = validation broken
    
    # Campaign tracking alerts
    UTM_MISSING_THRESHOLD = 20  # >20% leads missing UTM params
    SFMC_SYNC_DELAY_HOURS = 4  # SFMC sync should happen within 4 hours
    
    # Data quality alerts
    INVALID_EMAIL_THRESHOLD = 5  # >5% invalid emails = form issue
    LEAD_SCORE_ANOMALY_THRESHOLD = 2.5  # Std deviations from mean
    
    # Integration health
    API_FAILURE_THRESHOLD = 3  # 3 consecutive failures = system issue


# ============================================================================
# SYNTHETIC DATA GENERATION - Simulates Real SFDC/SFMC Data
# ============================================================================

def generate_campaign_performance_data(days=30):
    """
    Generate realistic campaign metrics with intentional anomalies
    that mirror real marketing ops issues.
    """
    
    campaigns = [
        'Product_Launch_Email',
        'Nurture_Sequence_Week1',
        'Webinar_Invite',
        'Case_Study_Download',
        'Trial_Reminder'
    ]
    
    records = []
    
    for day in range(days):
        date = datetime.now() - timedelta(days=days-day)
        
        for campaign in campaigns:
            # Baseline metrics (healthy state)
            baseline_sends = random.randint(800, 1200)
            baseline_deliverability = random.uniform(0.96, 0.99)
            baseline_opens = random.uniform(0.18, 0.25)
            baseline_clicks = random.uniform(0.03, 0.06)
            baseline_unsubscribes = random.uniform(0.001, 0.003)
            baseline_bounces = random.uniform(0.01, 0.03)
            
            # INJECT REALISTIC ANOMALIES
            
            # Anomaly 1: API failure on day 22 (common issue)
            if day == 22 and campaign == 'Product_Launch_Email':
                baseline_sends = int(baseline_sends * 0.3)  # Only 30% sent
                baseline_deliverability = 0.60  # Major drop
            
            # Anomaly 2: Deliverability drop on day 18 (ISP blacklist)
            if day >= 18 and day <= 20 and campaign == 'Nurture_Sequence_Week1':
                baseline_deliverability = random.uniform(0.75, 0.82)
                baseline_bounces = random.uniform(0.08, 0.12)
            
            # Anomaly 3: Unsubscribe spike on day 15 (wrong audience)
            if day == 15 and campaign == 'Webinar_Invite':
                baseline_unsubscribes = random.uniform(0.015, 0.025)  # 10x normal
            
            # Anomaly 4: Tracking broken on day 10 (UTM params missing)
            if day >= 10 and day <= 12 and campaign == 'Case_Study_Download':
                baseline_clicks = baseline_clicks * 0.4  # Appear lower due to tracking
            
            # Calculate actual metrics
            total_sends = int(baseline_sends)
            delivered = int(total_sends * baseline_deliverability)
            bounced = total_sends - delivered
            opens = int(delivered * baseline_opens)
            clicks = int(opens * (baseline_clicks / baseline_opens))
            unsubscribes = int(delivered * baseline_unsubscribes)
            
            records.append({
                'date': date.date(),
                'campaign_name': campaign,
                'sends': total_sends,
                'delivered': delivered,
                'bounced': bounced,
                'opens': opens,
                'clicks': clicks,
                'unsubscribes': unsubscribes,
                'deliverability_rate': delivered / total_sends if total_sends > 0 else 0,
                'bounce_rate': bounced / total_sends if total_sends > 0 else 0,
                'open_rate': opens / delivered if delivered > 0 else 0,
                'click_rate': clicks / delivered if delivered > 0 else 0,
                'unsubscribe_rate': unsubscribes / delivered if delivered > 0 else 0
            })
    
    return pd.DataFrame(records)


def generate_lead_flow_data(days=30):
    """
    Generate lead capture metrics with common failure modes.
    """
    
    sources = ['Organic', 'Paid Search', 'LinkedIn', 'Webinar', 'Content']
    
    records = []
    
    for day in range(days):
        date = datetime.now() - timedelta(days=days-day)
        
        for source in sources:
            # Baseline
            baseline_form_loads = random.randint(200, 400)
            baseline_submissions = int(baseline_form_loads * random.uniform(0.35, 0.45))
            baseline_valid_emails = baseline_submissions * random.uniform(0.95, 0.98)
            baseline_duplicates = int(baseline_submissions * random.uniform(0.02, 0.05))
            
            # INJECT ANOMALIES
            
            # Anomaly 1: Form broken on day 20
            if day == 20 and source == 'Paid Search':
                baseline_submissions = int(baseline_form_loads * 0.15)  # Massive drop
            
            # Anomaly 2: Lead volume crash on day 25
            if day >= 25 and source == 'Organic':
                baseline_form_loads = int(baseline_form_loads * 0.4)
                baseline_submissions = int(baseline_submissions * 0.4)
            
            # Anomaly 3: Validation broken on day 12
            if day >= 12 and day <= 14 and source == 'LinkedIn':
                baseline_valid_emails = baseline_submissions * 0.70  # 30% invalid!
            
            # Anomaly 4: Duplicate detection off on day 8
            if day >= 8 and day <= 10 and source == 'Webinar':
                baseline_duplicates = int(baseline_submissions * 0.18)  # 18% dupes
            
            # Calculate
            form_loads = baseline_form_loads
            submissions = int(baseline_submissions)
            valid_emails = int(baseline_valid_emails)
            duplicates = baseline_duplicates
            synced_to_sfmc = int(valid_emails * random.uniform(0.92, 0.99))
            
            # Anomaly 5: SFMC sync delay on day 27
            if day == 27:
                synced_to_sfmc = int(valid_emails * 0.65)  # Only 65% synced
            
            records.append({
                'date': date.date(),
                'source': source,
                'form_loads': form_loads,
                'form_submissions': submissions,
                'valid_emails': valid_emails,
                'invalid_emails': submissions - valid_emails,
                'duplicates': duplicates,
                'synced_to_sfmc': synced_to_sfmc,
                'conversion_rate': submissions / form_loads if form_loads > 0 else 0,
                'invalid_rate': (submissions - valid_emails) / submissions if submissions > 0 else 0,
                'duplicate_rate': duplicates / submissions if submissions > 0 else 0,
                'sync_rate': synced_to_sfmc / valid_emails if valid_emails > 0 else 0
            })
    
    return pd.DataFrame(records)


def generate_lead_scoring_data(days=30):
    """
    Lead scoring health - detect when model goes haywire.
    """
    
    records = []
    
    for day in range(days):
        date = datetime.now() - timedelta(days=days-day)
        
        # Normal distribution: mean=45, std=15
        num_leads = random.randint(80, 150)
        
        # Anomaly: Scoring model broken on day 16
        if day >= 16 and day <= 18:
            # All leads getting same score = model broken
            scores = [random.randint(20, 30) for _ in range(num_leads)]
        else:
            # Normal distribution
            scores = np.random.normal(45, 15, num_leads)
            scores = [max(0, min(100, s)) for s in scores]
        
        records.append({
            'date': date.date(),
            'num_leads': num_leads,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'scores_distribution': scores
        })
    
    return pd.DataFrame(records)


# ============================================================================
# ANOMALY DETECTION ALGORITHMS
# ============================================================================

def detect_deliverability_anomalies(campaign_df):
    """
    Catch email deliverability issues before they impact reputation.
    """
    
    anomalies = []
    
    for campaign in campaign_df['campaign_name'].unique():
        campaign_data = campaign_df[campaign_df['campaign_name'] == campaign].sort_values('date')
        
        # Calculate rolling 7-day baseline
        campaign_data['deliverability_baseline'] = campaign_data['deliverability_rate'].rolling(7, min_periods=3).mean()
        
        for idx, row in campaign_data.iterrows():
            if pd.isna(row['deliverability_baseline']):
                continue
            
            # Check for drops
            pct_drop = ((row['deliverability_baseline'] - row['deliverability_rate']) / row['deliverability_baseline']) * 100
            
            if pct_drop > OpsConfig.DELIVERABILITY_DROP_THRESHOLD:
                anomalies.append({
                    'date': row['date'],
                    'campaign': campaign,
                    'anomaly_type': 'Deliverability Drop',
                    'severity': 'HIGH' if pct_drop > 25 else 'MEDIUM',
                    'current_value': f"{row['deliverability_rate']*100:.1f}%",
                    'baseline_value': f"{row['deliverability_baseline']*100:.1f}%",
                    'description': f"Deliverability dropped {pct_drop:.1f}% below baseline. Possible ISP block or API issue.",
                    'action': 'Check: 1) Salesforce API logs, 2) SFMC sending IP reputation, 3) Bounce reasons'
                })
            
            # Check bounce rate
            if row['bounce_rate'] > (OpsConfig.BOUNCE_RATE_THRESHOLD / 100):
                anomalies.append({
                    'date': row['date'],
                    'campaign': campaign,
                    'anomaly_type': 'High Bounce Rate',
                    'severity': 'HIGH',
                    'current_value': f"{row['bounce_rate']*100:.1f}%",
                    'baseline_value': f"<{OpsConfig.BOUNCE_RATE_THRESHOLD}%",
                    'description': f"Bounce rate at {row['bounce_rate']*100:.1f}% (threshold: {OpsConfig.BOUNCE_RATE_THRESHOLD}%). List hygiene issue.",
                    'action': 'Check: 1) List source, 2) Email validation, 3) Recent imports'
                })
    
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


def detect_lead_flow_anomalies(lead_df):
    """
    Catch lead capture and sync issues.
    """
    
    anomalies = []
    
    for source in lead_df['source'].unique():
        source_data = lead_df[lead_df['source'] == source].sort_values('date')
        
        # Baseline calculations
        source_data['submissions_baseline'] = source_data['form_submissions'].rolling(7, min_periods=3).mean()
        source_data['conversion_baseline'] = source_data['conversion_rate'].rolling(7, min_periods=3).mean()
        
        for idx, row in source_data.iterrows():
            if pd.isna(row['submissions_baseline']):
                continue
            
            # Lead volume drop
            pct_drop = ((row['submissions_baseline'] - row['form_submissions']) / row['submissions_baseline']) * 100
            
            if pct_drop > OpsConfig.LEAD_VOLUME_DROP_THRESHOLD:
                anomalies.append({
                    'date': row['date'],
                    'source': source,
                    'anomaly_type': 'Lead Volume Drop',
                    'severity': 'HIGH',
                    'current_value': f"{row['form_submissions']} leads",
                    'baseline_value': f"{row['submissions_baseline']:.0f} leads",
                    'description': f"Lead volume dropped {pct_drop:.1f}% below baseline. Possible traffic or form issue.",
                    'action': 'Check: 1) Ad campaigns still running, 2) Form functionality, 3) Website errors'
                })
            
            # Form conversion drop (form broken)
            if not pd.isna(row['conversion_baseline']):
                conversion_drop = ((row['conversion_baseline'] - row['conversion_rate']) / row['conversion_baseline']) * 100
                
                if conversion_drop > 50:  # >50% drop in form conversion
                    anomalies.append({
                        'date': row['date'],
                        'source': source,
                        'anomaly_type': 'Form Conversion Crash',
                        'severity': 'CRITICAL',
                        'current_value': f"{row['conversion_rate']*100:.1f}%",
                        'baseline_value': f"{row['conversion_baseline']*100:.1f}%",
                        'description': f"Form conversion dropped {conversion_drop:.1f}%. Form likely broken.",
                        'action': 'CHECK IMMEDIATELY: 1) Test form submission, 2) Check JavaScript errors, 3) Review recent deploys'
                    })
            
            # Invalid email spike
            if row['invalid_rate'] > (OpsConfig.INVALID_EMAIL_THRESHOLD / 100):
                anomalies.append({
                    'date': row['date'],
                    'source': source,
                    'anomaly_type': 'Invalid Email Spike',
                    'severity': 'MEDIUM',
                    'current_value': f"{row['invalid_rate']*100:.1f}%",
                    'baseline_value': f"<{OpsConfig.INVALID_EMAIL_THRESHOLD}%",
                    'description': f"Invalid email rate at {row['invalid_rate']*100:.1f}%. Email validation broken.",
                    'action': 'Check: 1) Form validation rules, 2) API payload, 3) Recent form changes'
                })
            
            # Duplicate spike
            if row['duplicate_rate'] > (OpsConfig.DUPLICATE_LEAD_THRESHOLD / 100):
                anomalies.append({
                    'date': row['date'],
                    'source': source,
                    'anomaly_type': 'Duplicate Lead Spike',
                    'severity': 'MEDIUM',
                    'current_value': f"{row['duplicate_rate']*100:.1f}%",
                    'baseline_value': f"<{OpsConfig.DUPLICATE_LEAD_THRESHOLD}%",
                    'description': f"Duplicate rate at {row['duplicate_rate']*100:.1f}%. Deduplication broken.",
                    'action': 'Check: 1) SFDC duplicate rules, 2) Matching logic, 3) Integration settings'
                })
            
            # SFMC sync issue
            if row['sync_rate'] < 0.90:  # Less than 90% synced
                anomalies.append({
                    'date': row['date'],
                    'source': source,
                    'anomaly_type': 'SFMC Sync Delay',
                    'severity': 'HIGH',
                    'current_value': f"{row['sync_rate']*100:.1f}% synced",
                    'baseline_value': ">95% synced",
                    'description': f"Only {row['sync_rate']*100:.1f}% of leads synced to SFMC. Integration delay.",
                    'action': 'Check: 1) SFMC connector status, 2) API limits, 3) Field mapping errors'
                })
    
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


def detect_scoring_anomalies(scoring_df):
    """
    Catch when lead scoring model goes sideways.
    """
    
    anomalies = []
    
    # Calculate overall baseline
    overall_mean = scoring_df['avg_score'].mean()
    overall_std = scoring_df['avg_score'].std()
    
    for idx, row in scoring_df.iterrows():
        # Z-score detection
        z_score = (row['avg_score'] - overall_mean) / overall_std if overall_std > 0 else 0
        
        if abs(z_score) > OpsConfig.LEAD_SCORE_ANOMALY_THRESHOLD:
            anomalies.append({
                'date': row['date'],
                'source': 'Lead Scoring Model',
                'anomaly_type': 'Scoring Model Drift',
                'severity': 'MEDIUM',
                'current_value': f"Avg: {row['avg_score']:.1f}",
                'baseline_value': f"Avg: {overall_mean:.1f}",
                'description': f"Average score deviated {abs(z_score):.1f} std devs from baseline. Model may be broken.",
                'action': 'Check: 1) Scoring rules still active, 2) Field mappings, 3) Recent config changes'
            })
        
        # Low variance = all leads getting same score (broken model)
        if row['std_score'] < 5:
            anomalies.append({
                'date': row['date'],
                'source': 'Lead Scoring Model',
                'anomaly_type': 'Scoring Model Flatlined',
                'severity': 'CRITICAL',
                'current_value': f"Std Dev: {row['std_score']:.1f}",
                'baseline_value': "Std Dev: ~15",
                'description': f"Score variance collapsed. All leads getting similar scores. Model likely broken.",
                'action': 'CHECK IMMEDIATELY: 1) Scoring rules, 2) Field calculations, 3) Recent workflow changes'
            })
    
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


# ============================================================================
# REPORTING & VISUALIZATION
# ============================================================================

def generate_anomaly_summary(all_anomalies_df):
    """
    Executive summary of issues found.
    """
    
    if all_anomalies_df.empty:
        return pd.DataFrame({'status': ['All systems healthy']})
    
    summary = {
        'metric': [
            'Total Anomalies Detected',
            'Critical Severity',
            'High Severity',
            'Medium Severity',
            'Most Common Issue',
            'Campaigns Affected',
            'Date Range'
        ],
        'value': [
            len(all_anomalies_df),
            len(all_anomalies_df[all_anomalies_df['severity'] == 'CRITICAL']),
            len(all_anomalies_df[all_anomalies_df['severity'] == 'HIGH']),
            len(all_anomalies_df[all_anomalies_df['severity'] == 'MEDIUM']),
            all_anomalies_df['anomaly_type'].value_counts().index[0] if not all_anomalies_df.empty else 'N/A',
            all_anomalies_df.get('campaign', all_anomalies_df.get('source', pd.Series())).nunique(),
            f"{all_anomalies_df['date'].min()} to {all_anomalies_df['date'].max()}"
        ]
    }
    
    return pd.DataFrame(summary)


def create_anomaly_timeline(all_anomalies_df, output_dir):
    """
    Visual timeline of when issues occurred.
    """
    
    if all_anomalies_df.empty:
        print("  ⚠ No anomalies to visualize")
        return
    
    # Count anomalies by date and severity
    timeline = all_anomalies_df.groupby(['date', 'severity']).size().reset_index(name='count')
    timeline_pivot = timeline.pivot(index='date', columns='severity', values='count').fillna(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'CRITICAL': '#E63946', 'HIGH': '#F77F00', 'MEDIUM': '#FCBF49'}
    
    timeline_pivot.plot(kind='bar', stacked=True, color=[colors.get(c, '#999') for c in timeline_pivot.columns], ax=ax)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Anomalies', fontsize=12)
    ax.set_title('Marketing Operations Anomaly Timeline', fontsize=14, fontweight='bold')
    ax.legend(title='Severity', loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'anomaly_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Timeline chart saved")


def create_severity_breakdown(all_anomalies_df, output_dir):
    """
    Pie chart of issue types and severity.
    """
    
    if all_anomalies_df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # By anomaly type
    type_counts = all_anomalies_df['anomaly_type'].value_counts()
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Issues by Type', fontsize=12, fontweight='bold')
    
    # By severity
    severity_counts = all_anomalies_df['severity'].value_counts()
    colors_sev = [{'CRITICAL': '#E63946', 'HIGH': '#F77F00', 'MEDIUM': '#FCBF49'}.get(s, '#999') for s in severity_counts.index]
    ax2.pie(severity_counts.values, labels=severity_counts.index, colors=colors_sev, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Issues by Severity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Severity breakdown chart saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("=" * 70)
    print("MARKETING OPERATIONS ANOMALY DETECTOR")
    print("Proactive Issue Detection for Salesforce + Marketing Cloud")
    print("=" * 70)
    print()
    
    # Setup
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'
    alerts_dir = base_dir / 'alerts'
    
    for directory in [data_dir, output_dir, alerts_dir]:
        directory.mkdir(exist_ok=True)
    
    # ========================================
    # STEP 1: GENERATE DATA
    # ========================================
    print("📊 Generating synthetic marketing operations data...")
    
    campaign_data = generate_campaign_performance_data(days=30)
    lead_flow_data = generate_lead_flow_data(days=30)
    scoring_data = generate_lead_scoring_data(days=30)
    
    print(f"  ✓ Generated {len(campaign_data)} campaign performance records")
    print(f"  ✓ Generated {len(lead_flow_data)} lead flow records")
    print(f"  ✓ Generated {len(scoring_data)} scoring health records")
    print()
    
    # Save raw data
    campaign_data.to_csv(data_dir / 'campaign_performance.csv', index=False)
    lead_flow_data.to_csv(data_dir / 'lead_flow.csv', index=False)
    scoring_data.to_csv(data_dir / 'lead_scoring.csv', index=False)
    
    # ========================================
    # STEP 2: DETECT ANOMALIES
    # ========================================
    print("🔍 Running anomaly detection algorithms...")
    
    deliverability_anomalies = detect_deliverability_anomalies(campaign_data)
    lead_flow_anomalies = detect_lead_flow_anomalies(lead_flow_data)
    scoring_anomalies = detect_scoring_anomalies(scoring_data)
    
    print(f"  ✓ Found {len(deliverability_anomalies)} deliverability issues")
    print(f"  ✓ Found {len(lead_flow_anomalies)} lead flow issues")
    print(f"  ✓ Found {len(scoring_anomalies)} scoring model issues")
    print()
    
    # Combine all anomalies
    all_anomalies = pd.concat([
        deliverability_anomalies,
        lead_flow_anomalies,
        scoring_anomalies
    ], ignore_index=True)
    
    # ========================================
    # STEP 3: PRIORITIZE & REPORT
    # ========================================
    print("📋 Generating reports...")
    
    # Sort by severity and date
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
    all_anomalies['severity_rank'] = all_anomalies['severity'].map(severity_order)
    all_anomalies = all_anomalies.sort_values(['severity_rank', 'date'], ascending=[True, False])
    all_anomalies = all_anomalies.drop('severity_rank', axis=1)
    
    # Generate summary
    summary = generate_anomaly_summary(all_anomalies)
    
    # Export
    all_anomalies.to_csv(alerts_dir / 'all_anomalies.csv', index=False)
    summary.to_csv(output_dir / 'anomaly_summary.csv', index=False)
    
    # Export critical alerts separately
    critical_alerts = all_anomalies[all_anomalies['severity'] == 'CRITICAL']
    if not critical_alerts.empty:
        critical_alerts.to_csv(alerts_dir / 'CRITICAL_ALERTS.csv', index=False)
        print(f"  ⚠️  {len(critical_alerts)} CRITICAL alerts exported")
    
    print(f"  ✓ Exported: all_anomalies.csv")
    print(f"  ✓ Exported: anomaly_summary.csv")
    print()
    
    # ========================================
    # STEP 4: VISUALIZATIONS
    # ========================================
    print("📊 Creating visualizations...")
    
    create_anomaly_timeline(all_anomalies, output_dir)
    create_severity_breakdown(all_anomalies, output_dir)
    print()
    
    # ========================================
    # SUMMARY
    # ========================================
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    
    if all_anomalies.empty:
        print("✅ All systems healthy - no anomalies detected!")
    else:
        print(f"📌 TOTAL ISSUES DETECTED: {len(all_anomalies)}")
        print()
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM']:
            severity_issues = all_anomalies[all_anomalies['severity'] == severity]
            if not severity_issues.empty:
                print(f"🔴 {severity} SEVERITY ({len(severity_issues)} issues):")
                for _, issue in severity_issues.head(3).iterrows():
                    print(f"   • {issue['date']}: {issue['anomaly_type']}")
                    print(f"     {issue['description']}")
                    print(f"     → {issue['action']}")
                    print()
    
    print("=" * 70)
    print("✅ Analysis complete! Check /alerts for actionable issues.")
    print("=" * 70)


if __name__ == "__main__":
    main()
