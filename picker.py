#!/usr/bin/env python3
"""
Manual P-Wave Picking Tool for Seismic Waveforms
Allows manual review and picking of P-arrivals with PhaseNet reference
"""

import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from obspy import UTCDateTime
import pandas as pd
from typing import Optional, Tuple
import sys

class ManualPicker:
    def __init__(self, db_path: str = 'seismic_data.db'):
        """Initialize the manual picking tool"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Add manual pick column if it doesn't exist
        self._add_manual_pick_column()
        
        # Current waveform index
        self.current_idx = 0
        self.waveforms = []
        self.current_waveform = None
        self.manual_pick = None
        
        # Plot elements
        self.fig = None
        self.ax = None
        self.pick_line = None
        self.phasenet_line = None
        
        # Zoom/pan state
        self.toolbar = None
        
    def _add_manual_pick_column(self):
        """Add manual_p_pick_time column if it doesn't exist"""
        try:
            self.cursor.execute('''
                ALTER TABLE waveforms 
                ADD COLUMN manual_p_pick_time REAL
            ''')
            self.conn.commit()
            print("Added manual_p_pick_time column to database")
        except sqlite3.OperationalError:
            # Column already exists
            pass
    
    def load_waveforms(self, skip_picked: bool = True):
        """
        Load all waveforms from database
        
        Parameters:
        -----------
        skip_picked : bool
            Skip waveforms that already have manual picks (default: True)
        """
        
        # Build query to get all waveforms
        query = '''
            SELECT w.id, w.earthquake_id, w.station_code, w.network, w.channel,
                   w.waveform_data, w.sampling_rate, w.p_pick_time, w.manual_p_pick_time,
                   w.eq_time, w.pre_time, w.post_time, w.distance_km,
                   e.magnitude, e.latitude, e.longitude, e.place, e.depth
            FROM waveforms w
            JOIN earthquakes e ON w.earthquake_id = e.id
        '''
        
        if skip_picked:
            query += ' WHERE w.manual_p_pick_time IS NULL'
        
        query += ' ORDER BY e.event_time, w.distance_km'
        
        # Execute query
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Parse results
        self.waveforms = []
        for row in results:
            waveform_dict = {
                'id': row[0],
                'earthquake_id': row[1],
                'station_code': row[2],
                'network': row[3],
                'channel': row[4],
                'waveform_data': pickle.loads(row[5]),
                'sampling_rate': row[6],
                'p_pick_time': row[7],
                'manual_p_pick_time': row[8],
                'eq_time': row[9],
                'pre_time': row[10],
                'post_time': row[11],
                'distance_km': row[12],
                'magnitude': row[13],
                'eq_lat': row[14],
                'eq_lon': row[15],
                'place': row[16],
                'eq_depth': row[17]
            }
            self.waveforms.append(waveform_dict)
        
        total_query = 'SELECT COUNT(*) FROM waveforms'
        self.cursor.execute(total_query)
        total_count = self.cursor.fetchone()[0]
        
        if skip_picked:
            picked_query = 'SELECT COUNT(*) FROM waveforms WHERE manual_p_pick_time IS NOT NULL'
            self.cursor.execute(picked_query)
            picked_count = self.cursor.fetchone()[0]
            print(f"Loaded {len(self.waveforms)} unpicked waveforms (skipped {picked_count} already picked, {total_count} total)")
        else:
            print(f"Loaded {len(self.waveforms)} waveforms total")
        
        return len(self.waveforms)
    
    def save_manual_pick(self):
        """Save the manual pick to the database"""
        if self.manual_pick is not None and self.current_waveform is not None:
            self.cursor.execute('''
                UPDATE waveforms 
                SET manual_p_pick_time = ?
                WHERE id = ?
            ''', (self.manual_pick, self.current_waveform['id']))
            self.conn.commit()
            print(f"  ✓ Saved manual pick for {self.current_waveform['network']}.{self.current_waveform['station_code']}")
    
    def plot_waveform(self):
        """Plot the current waveform with PhaseNet pick reference"""
        if not self.waveforms:
            print("No waveforms loaded!")
            return
        
        self.current_waveform = self.waveforms[self.current_idx]
        wf = self.current_waveform
        
        # Clear previous plot
        self.ax.clear()
        
        # Create time axis
        npts = len(wf['waveform_data'])
        dt = 1.0 / wf['sampling_rate']
        time = np.arange(npts) * dt - wf['pre_time']  # Time relative to earthquake
        
        # Use raw waveform data
        raw_data = wf['waveform_data'].copy()
        
        # Normalize for display
        max_amp = np.max(np.abs(raw_data))
        if max_amp > 0:
            raw_data = raw_data / max_amp
        
        # Plot waveform
        self.ax.plot(time, raw_data, 'k-', linewidth=0.5, alpha=0.8)
        
        # Plot earthquake origin time
        self.ax.axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='Origin')
        
        # Plot PhaseNet pick if available
        if wf['p_pick_time'] is not None:
            phasenet_time = wf['p_pick_time'] - wf['eq_time']
            self.phasenet_line = self.ax.axvline(phasenet_time, color='blue', 
                                                 linestyle='--', alpha=0.7, linewidth=1.5,
                                                 label=f'PhaseNet: {phasenet_time:.2f}s')
        
        # Plot existing manual pick if available
        if wf['manual_p_pick_time'] is not None:
            manual_time = wf['manual_p_pick_time'] - wf['eq_time']
            self.ax.axvline(manual_time, color='green', linestyle='-', 
                           alpha=0.7, linewidth=2,
                           label=f'Manual: {manual_time:.2f}s')
        
        # Labels and title
        self.ax.set_xlabel('Time relative to origin (s)', fontsize=11)
        self.ax.set_ylabel('Normalized amplitude', fontsize=11)
        
        # Create informative title with station and earthquake info
        title_lines = []
        title_lines.append(f"{wf['network']}.{wf['station_code']}.{wf['channel']} | "
                          f"Distance: {wf['distance_km']:.1f} km | "
                          f"Waveform {self.current_idx + 1} of {len(self.waveforms)}")
        
        eq_info = f"M{wf['magnitude']:.1f}"
        if wf['place']:
            eq_info += f" - {wf['place']}"
        eq_info += f" | Depth: {wf['eq_depth']:.1f} km"
        title_lines.append(eq_info)
        
        if wf['p_pick_time'] is not None:
            phasenet_time = wf['p_pick_time'] - wf['eq_time']
            expected_time = wf['distance_km'] / 8.0  # Rough P-wave velocity
            title_lines.append(f"Expected P-arrival: ~{expected_time:.1f}s | PhaseNet pick: {phasenet_time:.2f}s")
        
        self.ax.set_title('\n'.join(title_lines), fontsize=10)
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        
        # Set initial x-limits (can be zoomed/panned by user)
        self.ax.set_xlim([-wf['pre_time'], min(60, wf['post_time'])])
        self.ax.set_ylim([-1.1, 1.1])
        
        # Reset pick line reference
        self.pick_line = None
        
        plt.draw()
    
    def on_click(self, event):
        """Handle mouse click for picking"""
        if event.inaxes != self.ax:
            return
        
        # Only pick with left click and when not in zoom/pan mode
        if event.button == 1 and self.toolbar.mode == '':
            # Remove previous pick line if exists
            if self.pick_line is not None:
                self.pick_line.remove()
            
            # Store the pick time (relative to earthquake origin)
            pick_time_rel = event.xdata
            self.manual_pick = self.current_waveform['eq_time'] + pick_time_rel
            
            # Draw the pick line
            self.pick_line = self.ax.axvline(pick_time_rel, color='green', 
                                            linewidth=2, alpha=0.8,
                                            label=f'New pick: {pick_time_rel:.2f}s')
            
            # Update legend
            handles, labels = self.ax.get_legend_handles_labels()
            # Remove old "New pick" entry if exists
            filtered_handles = []
            filtered_labels = []
            for h, l in zip(handles, labels):
                if not l.startswith('New pick'):
                    filtered_handles.append(h)
                    filtered_labels.append(l)
            filtered_handles.append(self.pick_line)
            filtered_labels.append(f'New pick: {pick_time_rel:.2f}s')
            self.ax.legend(filtered_handles, filtered_labels, loc='upper right', fontsize=9)
            
            plt.draw()
            
            print(f"  → Picked at {pick_time_rel:.2f}s relative to origin")
    
    def next_waveform(self, event=None):
        """Move to next waveform"""
        # Save current pick if exists
        if self.manual_pick is not None:
            self.save_manual_pick()
        
        # Move to next
        if self.current_idx < len(self.waveforms) - 1:
            self.current_idx += 1
            self.manual_pick = None
            self.pick_line = None
            self.plot_waveform()
        else:
            print("\n*** Reached end of waveforms ***")
            self.save_and_quit()
    
    def previous_waveform(self, event=None):
        """Move to previous waveform"""
        # Save current pick if exists
        if self.manual_pick is not None:
            self.save_manual_pick()
        
        # Move to previous
        if self.current_idx > 0:
            self.current_idx -= 1
            self.manual_pick = None
            self.pick_line = None
            self.plot_waveform()
        else:
            print("Already at first waveform")
    
    def skip_waveform(self, event=None):
        """Skip current waveform without picking (for bad/unclear waveforms)"""
        print(f"  ✗ Skipped {self.current_waveform['network']}.{self.current_waveform['station_code']} (bad/unclear waveform)")
        self.manual_pick = None
        self.next_waveform()

    def accept_phasenet_pick(self, event=None):
        """Accept the PhaseNet pick as the manual pick"""
        if self.current_waveform is None:
            return

        wf = self.current_waveform
        if wf['p_pick_time'] is None:
            print("  ⚠ No PhaseNet pick available for this waveform")
            return

        # Remove previous pick line if exists
        if self.pick_line is not None:
            self.pick_line.remove()

        # Set the manual pick to the PhaseNet pick time
        self.manual_pick = wf['p_pick_time']
        phasenet_time_rel = wf['p_pick_time'] - wf['eq_time']

        # Draw the pick line (green for accepted PhaseNet pick)
        self.pick_line = self.ax.axvline(phasenet_time_rel, color='green',
                                        linewidth=2, alpha=0.8,
                                        label=f'Accepted: {phasenet_time_rel:.2f}s')

        # Update legend
        handles, labels = self.ax.get_legend_handles_labels()
        # Remove old "New pick" or "Accepted" entry if exists
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if not l.startswith('New pick') and not l.startswith('Accepted'):
                filtered_handles.append(h)
                filtered_labels.append(l)
        filtered_handles.append(self.pick_line)
        filtered_labels.append(f'Accepted: {phasenet_time_rel:.2f}s')
        self.ax.legend(filtered_handles, filtered_labels, loc='upper right', fontsize=9)

        plt.draw()

        print(f"  ✓ Accepted PhaseNet pick at {phasenet_time_rel:.2f}s relative to origin")
    
    def save_and_quit(self, event=None):
        """Save current pick and quit"""
        if self.manual_pick is not None:
            self.save_manual_pick()
        
        # Print summary
        picked_query = 'SELECT COUNT(*) FROM waveforms WHERE manual_p_pick_time IS NOT NULL'
        self.cursor.execute(picked_query)
        total_picked = self.cursor.fetchone()[0]
        
        print("\n" + "="*60)
        print(f"Picking session completed!")
        print(f"Total manual picks in database: {total_picked}")
        print("="*60)
        
        self.conn.close()
        plt.close('all')
        sys.exit(0)
    
    def run(self):
        """Run the manual picking interface"""
        if not self.waveforms:
            print("No waveforms to pick!")
            return
        
        # Create figure with toolbar
        self.fig = plt.figure(figsize=(15, 9))
        
        # Main plot
        self.ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.15, top=0.92)
        
        # Get the toolbar for zoom/pan functionality
        self.toolbar = self.fig.canvas.toolbar
        
        # Add navigation buttons
        button_width = 0.08
        button_height = 0.04
        button_y = 0.05

        ax_prev = plt.axes([0.25, button_y, button_width, button_height])
        ax_accept = plt.axes([0.34, button_y, button_width, button_height])
        ax_skip = plt.axes([0.43, button_y, button_width, button_height])
        ax_next = plt.axes([0.52, button_y, button_width, button_height])
        ax_quit = plt.axes([0.61, button_y, button_width, button_height])

        btn_prev = Button(ax_prev, '← Previous', color='lightblue')
        btn_accept = Button(ax_accept, 'Accept PN', color='lightcyan')
        btn_skip = Button(ax_skip, 'Skip (Bad)', color='lightyellow')
        btn_next = Button(ax_next, 'Next →', color='lightgreen')
        btn_quit = Button(ax_quit, 'Save & Quit', color='lightcoral')

        btn_prev.on_clicked(self.previous_waveform)
        btn_accept.on_clicked(self.accept_phasenet_pick)
        btn_skip.on_clicked(self.skip_waveform)
        btn_next.on_clicked(self.next_waveform)
        btn_quit.on_clicked(self.save_and_quit)
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Connect keyboard shortcuts
        def on_key(event):
            if event.key == 'n' or event.key == 'right':
                self.next_waveform()
            elif event.key == 'p' or event.key == 'left':
                self.previous_waveform()
            elif event.key == 's':
                self.skip_waveform()
            elif event.key == 'a':
                self.accept_phasenet_pick()
            elif event.key == 'q':
                self.save_and_quit()
            elif event.key == 'r':
                # Reset zoom
                self.plot_waveform()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Plot first waveform
        self.plot_waveform()
        
        # Instructions
        print("\n" + "="*80)
        print("MANUAL P-WAVE PICKING TOOL")
        print("="*80)
        print("\nInstructions:")
        print("  • Use ZOOM/PAN tools in the toolbar to examine waveform")
        print("  • LEFT CLICK on the plot to pick P-arrival (when not in zoom/pan mode)")
        print("  • Press 'a' to accept PhaseNet pick as manual pick")
        print("  • Press 'n' or RIGHT ARROW to save pick and move to next")
        print("  • Press 'p' or LEFT ARROW to go back")
        print("  • Press 's' to skip bad/unclear waveforms")
        print("  • Press 'r' to reset zoom")
        print("  • Press 'q' to save and quit")
        print("\nReference lines:")
        print("  • RED dotted = Earthquake origin time")
        print("  • BLUE dashed = PhaseNet automatic pick")
        print("  • GREEN solid = Your manual pick")
        print("\nDisplaying: Raw waveform (normalized)")
        print("="*80 + "\n")
        
        plt.show()


def main():
    """Main function to run the manual picker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual P-wave picking tool')
    parser.add_argument('--db', type=str, default='seismic_data.db',
                       help='Path to database file')
    parser.add_argument('--include-picked', action='store_true',
                       help='Include already picked waveforms for review/re-picking')
    
    args = parser.parse_args()
    
    # Create picker instance
    picker = ManualPicker(args.db)
    
    # Load waveforms (skip already picked by default)
    n_waveforms = picker.load_waveforms(skip_picked=not args.include_picked)
    
    if n_waveforms > 0:
        # Run the picker
        picker.run()
    else:
        print("\n*** No unpicked waveforms found! ***")
        print("Use --include-picked to review already picked waveforms")


if __name__ == "__main__":
    main()