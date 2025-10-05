#!/usr/bin/env python3
"""
Database Analysis Tool for Seismic Picker
Analyzes pick statistics and generates example plots for manual vs PhaseNet picks
"""

import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import argparse
from typing import Optional, Dict, List, Tuple

class DatabaseAnalyzer:
    def __init__(self, db_path: str):
        """Initialize the database analyzer"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Verify database structure
        self._verify_database()

        # Create output directory
        db_name = Path(db_path).stem
        self.output_dir = Path(f"analyze_db_{db_name}")
        self.plots_dir = self.output_dir / "example_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _verify_database(self):
        """Verify that the database has the expected tables and columns"""
        try:
            # Check if tables exist
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in self.cursor.fetchall()]

            if 'waveforms' not in tables or 'earthquakes' not in tables:
                raise ValueError("Database missing required tables (waveforms, earthquakes)")

            # Check if manual_p_pick_time column exists
            self.cursor.execute("PRAGMA table_info(waveforms)")
            columns = [row[1] for row in self.cursor.fetchall()]

            if 'manual_p_pick_time' not in columns:
                print("Adding manual_p_pick_time column to database...")
                self.cursor.execute('''
                    ALTER TABLE waveforms
                    ADD COLUMN manual_p_pick_time REAL
                ''')
                self.conn.commit()

        except Exception as e:
            raise ValueError(f"Database verification failed: {e}")

    def analyze_pick_statistics(self) -> Dict:
        """Analyze pick statistics in the database"""
        print("Analyzing pick statistics...")

        # Total waveforms
        self.cursor.execute("SELECT COUNT(*) FROM waveforms")
        total_waveforms = self.cursor.fetchone()[0]

        # PhaseNet picks available
        self.cursor.execute("SELECT COUNT(*) FROM waveforms WHERE p_pick_time IS NOT NULL")
        phasenet_picks = self.cursor.fetchone()[0]

        # Manual picks available
        self.cursor.execute("SELECT COUNT(*) FROM waveforms WHERE manual_p_pick_time IS NOT NULL")
        manual_picks = self.cursor.fetchone()[0]

        # Both picks available
        self.cursor.execute("""
            SELECT COUNT(*) FROM waveforms
            WHERE p_pick_time IS NOT NULL AND manual_p_pick_time IS NOT NULL
        """)
        both_picks = self.cursor.fetchone()[0]

        # Agreement analysis (picks within tolerance)
        tolerance_s = 0.5  # 0.5 second tolerance
        self.cursor.execute("""
            SELECT COUNT(*) FROM waveforms
            WHERE p_pick_time IS NOT NULL AND manual_p_pick_time IS NOT NULL
            AND ABS(p_pick_time - manual_p_pick_time) <= ?
        """, (tolerance_s,))
        picks_agree = self.cursor.fetchone()[0]

        # Earthquakes info
        self.cursor.execute("SELECT COUNT(*) FROM earthquakes")
        total_earthquakes = self.cursor.fetchone()[0]

        # Station statistics
        self.cursor.execute("SELECT COUNT(DISTINCT station_code) FROM waveforms")
        unique_stations = self.cursor.fetchone()[0]

        stats = {
            'total_waveforms': total_waveforms,
            'total_earthquakes': total_earthquakes,
            'unique_stations': unique_stations,
            'phasenet_picks': phasenet_picks,
            'manual_picks': manual_picks,
            'both_picks': both_picks,
            'picks_agree': picks_agree,
            'tolerance_s': tolerance_s,
            'phasenet_coverage': phasenet_picks / total_waveforms * 100 if total_waveforms > 0 else 0,
            'manual_coverage': manual_picks / total_waveforms * 100 if total_waveforms > 0 else 0,
            'agreement_rate': picks_agree / both_picks * 100 if both_picks > 0 else 0,
        }

        return stats

    def get_example_waveforms(self) -> Dict[str, List[Dict]]:
        """Get example waveforms for each category"""
        print("Collecting example waveforms...")

        categories = {
            'same_picks': [],      # Manual and PhaseNet picks agree
            'different_picks': [], # Manual and PhaseNet picks differ
            'no_manual': [],       # Only PhaseNet pick (no manual)
            'manual_only': []      # Only manual pick (no PhaseNet)
        }

        # Same picks (within tolerance)
        tolerance_s = 0.5
        self.cursor.execute("""
            SELECT w.*, e.magnitude, e.latitude, e.longitude, e.place, e.depth
            FROM waveforms w
            JOIN earthquakes e ON w.earthquake_id = e.id
            WHERE w.p_pick_time IS NOT NULL AND w.manual_p_pick_time IS NOT NULL
            AND ABS(w.p_pick_time - w.manual_p_pick_time) <= ?
            ORDER BY RANDOM()
            LIMIT 5
        """, (tolerance_s,))

        for row in self.cursor.fetchall():
            categories['same_picks'].append(self._row_to_dict(row))

        # Different picks
        self.cursor.execute("""
            SELECT w.*, e.magnitude, e.latitude, e.longitude, e.place, e.depth
            FROM waveforms w
            JOIN earthquakes e ON w.earthquake_id = e.id
            WHERE w.p_pick_time IS NOT NULL AND w.manual_p_pick_time IS NOT NULL
            AND ABS(w.p_pick_time - w.manual_p_pick_time) > ?
            ORDER BY RANDOM()
            LIMIT 5
        """, (tolerance_s,))

        for row in self.cursor.fetchall():
            categories['different_picks'].append(self._row_to_dict(row))

        # No manual picks (PhaseNet only)
        self.cursor.execute("""
            SELECT w.*, e.magnitude, e.latitude, e.longitude, e.place, e.depth
            FROM waveforms w
            JOIN earthquakes e ON w.earthquake_id = e.id
            WHERE w.p_pick_time IS NOT NULL AND w.manual_p_pick_time IS NULL
            ORDER BY RANDOM()
            LIMIT 5
        """)

        for row in self.cursor.fetchall():
            categories['no_manual'].append(self._row_to_dict(row))

        # Manual only (no PhaseNet)
        self.cursor.execute("""
            SELECT w.*, e.magnitude, e.latitude, e.longitude, e.place, e.depth
            FROM waveforms w
            JOIN earthquakes e ON w.earthquake_id = e.id
            WHERE w.p_pick_time IS NULL AND w.manual_p_pick_time IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 5
        """)

        for row in self.cursor.fetchall():
            categories['manual_only'].append(self._row_to_dict(row))

        return categories

    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary"""
        return {
            'id': row[0],
            'earthquake_id': row[1],
            'station_code': row[2],
            'network': row[3],
            'channel': row[4],
            'location': row[5],
            'waveform_data': pickle.loads(row[6]),
            'sampling_rate': row[7],
            'p_pick_time': row[8],
            'eq_time': row[9],
            'pre_time': row[10],
            'post_time': row[11],
            'distance_km': row[12],
            'manual_p_pick_time': row[13],
            'magnitude': row[14],
            'eq_lat': row[15],
            'eq_lon': row[16],
            'place': row[17],
            'eq_depth': row[18]
        }

    def plot_waveform_example(self, wf: Dict, category: str, idx: int):
        """Plot a single waveform example"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Create time axis
        npts = len(wf['waveform_data'])
        dt = 1.0 / wf['sampling_rate']
        time = np.arange(npts) * dt - wf['pre_time']  # Time relative to earthquake

        # Normalize waveform for display
        raw_data = wf['waveform_data'].copy()
        max_amp = np.max(np.abs(raw_data))
        if max_amp > 0:
            raw_data = raw_data / max_amp

        # Plot waveform
        ax.plot(time, raw_data, 'k-', linewidth=0.8, alpha=0.7, label='Waveform')

        # Plot earthquake origin time
        ax.axvline(0, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label='Origin')

        # Plot PhaseNet pick if available
        if wf['p_pick_time'] is not None:
            phasenet_time = wf['p_pick_time'] - wf['eq_time']
            ax.axvline(phasenet_time, color='blue', linestyle='--', alpha=0.8,
                      linewidth=2, label=f'PhaseNet: {phasenet_time:.2f}s')

        # Plot manual pick if available
        if wf['manual_p_pick_time'] is not None:
            manual_time = wf['manual_p_pick_time'] - wf['eq_time']
            ax.axvline(manual_time, color='green', linestyle='-', alpha=0.8,
                      linewidth=2, label=f'Manual: {manual_time:.2f}s')

        # Calculate time difference if both picks exist
        time_diff_text = ""
        if wf['p_pick_time'] is not None and wf['manual_p_pick_time'] is not None:
            time_diff = wf['manual_p_pick_time'] - wf['p_pick_time']
            time_diff_text = f" | �t = {time_diff:.2f}s"

        # Labels and title
        ax.set_xlabel('Time relative to origin (s)', fontsize=11)
        ax.set_ylabel('Normalized amplitude', fontsize=11)

        # Title with station and earthquake info
        title = f"{category.replace('_', ' ').title()} Example {idx + 1}\n"
        title += f"{wf['network']}.{wf['station_code']}.{wf['channel']} | "
        title += f"Distance: {wf['distance_km']:.1f} km | M{wf['magnitude']:.1f}"
        if wf['place']:
            title += f" - {wf['place'][:50]}..."
        title += time_diff_text

        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set reasonable x-limits
        ax.set_xlim([-wf['pre_time'], min(60, wf['post_time'])])
        ax.set_ylim([-1.1, 1.1])

        # Save plot
        filename = f"{category}_{idx+1:02d}_{wf['network']}.{wf['station_code']}.png"
        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        return filename

    def generate_summary_plot(self, stats: Dict):
        """Generate a summary statistics plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Coverage pie chart
        labels = ['PhaseNet only', 'Manual only', 'Both picks', 'No picks']
        phasenet_only = stats['phasenet_picks'] - stats['both_picks']
        manual_only = stats['manual_picks'] - stats['both_picks']
        both = stats['both_picks']
        no_picks = stats['total_waveforms'] - stats['phasenet_picks'] - manual_only

        sizes = [phasenet_only, manual_only, both, no_picks]
        colors = ['lightblue', 'lightgreen', 'gold', 'lightcoral']

        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Pick Coverage Distribution')

        # 2. Pick counts bar chart
        categories = ['Total\nWaveforms', 'PhaseNet\nPicks', 'Manual\nPicks', 'Both\nPicks']
        counts = [stats['total_waveforms'], stats['phasenet_picks'],
                 stats['manual_picks'], stats['both_picks']]

        bars = ax2.bar(categories, counts, color=['gray', 'blue', 'green', 'orange'])
        ax2.set_title('Pick Counts')
        ax2.set_ylabel('Number of Waveforms')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom')

        # 3. Agreement analysis
        if stats['both_picks'] > 0:
            agree_pct = stats['agreement_rate']
            disagree_pct = 100 - agree_pct

            ax3.pie([agree_pct, disagree_pct],
                   labels=[f'Agree\n(�{stats["tolerance_s"]}s)', 'Disagree'],
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax3.set_title('Manual vs PhaseNet Agreement')
        else:
            ax3.text(0.5, 0.5, 'No overlapping\npicks to compare',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Manual vs PhaseNet Agreement')

        # 4. Database summary
        ax4.axis('off')
        summary_text = f"""
Database Summary:
" Total Earthquakes: {stats['total_earthquakes']:,}
" Total Waveforms: {stats['total_waveforms']:,}
" Unique Stations: {stats['unique_stations']:,}

Pick Coverage:
" PhaseNet: {stats['phasenet_coverage']:.1f}%
" Manual: {stats['manual_coverage']:.1f}%

Agreement Rate: {stats['agreement_rate']:.1f}%
(within �{stats['tolerance_s']}s tolerance)
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(self.output_dir / "database_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

        return "database_summary.png"

    def run_analysis(self):
        """Run complete database analysis"""
        print(f"\n{'='*60}")
        print(f"SEISMIC DATABASE ANALYSIS")
        print(f"Database: {self.db_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # 1. Analyze statistics
        stats = self.analyze_pick_statistics()

        # 2. Print statistics
        self.print_statistics(stats)

        # 3. Generate summary plot
        print("\nGenerating summary plots...")
        summary_plot = self.generate_summary_plot(stats)
        print(f"   Saved: {summary_plot}")

        # 4. Get example waveforms
        examples = self.get_example_waveforms()

        # 5. Generate example plots
        print("\nGenerating example waveform plots...")
        total_plots = 0

        for category, waveforms in examples.items():
            if not waveforms:
                print(f"  � No examples found for category: {category}")
                continue

            print(f"  Plotting {category} examples...")
            for idx, wf in enumerate(waveforms):
                filename = self.plot_waveform_example(wf, category, idx)
                print(f"     Saved: {filename}")
                total_plots += 1

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total plots generated: {total_plots + 1}")
        print(f"Output directory: {self.output_dir}")
        print(f"Summary plot: {self.output_dir}/database_summary.png")
        print(f"Example plots: {self.plots_dir}/")
        print(f"{'='*60}\n")

        # Close database connection
        self.conn.close()

    def print_statistics(self, stats: Dict):
        """Print formatted statistics"""
        print(f"Database Statistics:")
        print(f"{'-'*40}")
        print(f"Total Earthquakes:     {stats['total_earthquakes']:,}")
        print(f"Total Waveforms:       {stats['total_waveforms']:,}")
        print(f"Unique Stations:       {stats['unique_stations']:,}")
        print()
        print(f"Pick Availability:")
        print(f"{'-'*40}")
        print(f"PhaseNet Picks:        {stats['phasenet_picks']:,} ({stats['phasenet_coverage']:.1f}%)")
        print(f"Manual Picks:          {stats['manual_picks']:,} ({stats['manual_coverage']:.1f}%)")
        print(f"Both Available:        {stats['both_picks']:,}")
        print()
        print(f"Pick Agreement:")
        print(f"{'-'*40}")
        if stats['both_picks'] > 0:
            print(f"Agreement Rate:        {stats['agreement_rate']:.1f}% (�{stats['tolerance_s']}s tolerance)")
            print(f"Agreeing Picks:        {stats['picks_agree']:,}/{stats['both_picks']:,}")
        else:
            print(f"Agreement Rate:        N/A (no overlapping picks)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze seismic picker database')
    parser.add_argument('db_path', type=str, help='Path to the database file')

    args = parser.parse_args()

    # Verify database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database file not found: {args.db_path}")
        sys.exit(1)

    try:
        # Create analyzer and run
        analyzer = DatabaseAnalyzer(args.db_path)
        analyzer.run_analysis()

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()