import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import os

# ================= COLOR SCHEME =================
COLORS = {
    'bg_primary': '#f0f7f4',
    'bg_secondary': '#d4e8e1',
    'accent_pink': '#ffb7c5',
    'accent_green': '#a8e6cf',
    'accent_green_dark': '#56ab91',
    'accent_blue': '#b8e2f0',
    'accent_purple': '#c3b1e6',
    'text_dark': '#1e3a2f',
    'stress_high': '#ff8a80',
    'stress_medium': '#ffb74d',
    'stress_low': '#b9f6ca',
    'card_bg': '#ffffff',
    'selected_bg': '#e3f2fd',
    'border': '#d0dcd0'
}

# ================= DSP FUNCTIONS =================
def apply_savgol_filter(signal, window=21, order=3):
    """Apply Savitzky-Golay filter to signal"""
    # Ensure window is odd and > order
    if window % 2 == 0:
        window += 1
    if window <= order:
        window = order + 2
    return savgol_filter(signal, window_length=window, polyorder=order)

def calculate_heart_rate(peaks, time):
    if len(peaks) < 2:
        return 0
    rr_intervals = np.diff(time[peaks])
    if len(rr_intervals) == 0 or np.mean(rr_intervals) == 0:
        return 0
    hr = 60 / np.mean(rr_intervals)
    return hr

def detect_stress(gsr):
    """Detect stress based on GSR changes"""
    if len(gsr) < 10:
        return "Unknown"
    
    # Calculate rate of change
    diff = np.diff(gsr)
    mean_diff = np.mean(np.abs(diff))
    
    # Calculate baseline and recent values
    baseline = np.mean(gsr[:10])
    recent = np.mean(gsr[-10:])
    
    if recent > baseline * 1.2 or mean_diff > 0.05:
        return "Stressed 😰"
    else:
        return "Relaxed 😊"

def calculate_stress_level(hr, gsr_mean):
    """Calculate stress level percentage"""
    hr_component = min(100, max(0, (hr - 60) * 2.5))
    gsr_component = min(100, gsr_mean * 10)
    stress = (hr_component * 0.6 + gsr_component * 0.4)
    return min(100, stress)

def safe_float_conversion(value, default=0.5):
    """Safely convert any value to float"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        # If it's a string, try to extract numbers
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point and minus
            numeric_str = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
            if numeric_str:
                try:
                    return float(numeric_str)
                except:
                    return default
        return default

# ================= MAIN APP =================
class StressMonitoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CalmMind - Stress Monitoring System")
        self.root.geometry("1400x800")
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Data storage
        self.dataset_path = None
        self.dataset_df = None
        self.ppg_data = None
        self.gsr_data = None
        self.time_data = np.linspace(0, 10, 1000)  # Default time array
        self.filtered_ppg = None
        self.filtered_gsr = None
        
        # Patient selection tracking
        self.selected_patient = None
        self.selected_patient_id = None
        self.patient_details = {}
        self.data_source = "Not loaded"
        
        # Create notebook for pages
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create pages
        self.create_home_page()
        self.create_dataset_page()
        self.create_patients_page()
        self.create_analysis_page()
        self.create_report_page()
    
    # ================= HOME PAGE =================
    def create_home_page(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='🏠 Home')
        
        # Main container
        main = tk.Frame(frame, bg=COLORS['bg_primary'])
        main.pack(expand=True, fill='both', padx=50, pady=50)
        
        # Title with decorative elements
        title_frame = tk.Frame(main, bg=COLORS['bg_primary'])
        title_frame.pack(pady=20)
        
        # Decorative circles
        decor_canvas = tk.Canvas(title_frame, width=200, height=60, 
                                bg=COLORS['bg_primary'], highlightthickness=0)
        decor_canvas.pack()
        decor_canvas.create_oval(20, 10, 50, 40, fill=COLORS['accent_pink'], outline='')
        decor_canvas.create_oval(80, 10, 110, 40, fill=COLORS['accent_green'], outline='')
        decor_canvas.create_oval(140, 10, 170, 40, fill=COLORS['accent_blue'], outline='')
        
        tk.Label(title_frame, text='🧘 CalmMind', 
                font=('Arial', 48, 'bold'),
                bg=COLORS['bg_primary'],
                fg=COLORS['text_dark']).pack()
        
        tk.Label(title_frame, text='DSP-Based Stress Monitoring System',
                font=('Arial', 16),
                bg=COLORS['bg_primary'],
                fg=COLORS['accent_green_dark']).pack()
        
        # Team Information Card
        info_frame = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=2)
        info_frame.pack(pady=30, ipadx=40, ipady=20)
        
        # Department
        tk.Label(info_frame, text="Department of Computer Engineering",
                font=('Arial', 16, 'bold'),
                bg=COLORS['card_bg'],
                fg=COLORS['accent_purple']).pack(pady=(0, 15))
        
        # Team Members in two columns
        members_frame = tk.Frame(info_frame, bg=COLORS['card_bg'])
        members_frame.pack()
        
        # Left column - Team Members
        left_col = tk.Frame(members_frame, bg=COLORS['card_bg'])
        left_col.pack(side='left', padx=30)
        
        tk.Label(left_col, text="👥 Team Members:",
                font=('Arial', 14, 'bold'),
                bg=COLORS['card_bg']).pack(anchor='w')
        tk.Label(left_col, text="• Syeda Memoona Urooj (23-CE-006)",
                font=('Arial', 12),
                bg=COLORS['card_bg']).pack(anchor='w', pady=2)
        tk.Label(left_col, text="• Areeba Khalique (23-CE-016)",
                font=('Arial', 12),
                bg=COLORS['card_bg']).pack(anchor='w', pady=2)
        
        # Right column - Supervisor
        right_col = tk.Frame(members_frame, bg=COLORS['card_bg'])
        right_col.pack(side='left', padx=30)
        
        tk.Label(right_col, text="👩‍🏫 Supervisor:",
                font=('Arial', 14, 'bold'),
                bg=COLORS['card_bg']).pack(anchor='w')
        tk.Label(right_col, text="• Mam Hareem",
                font=('Arial', 12),
                bg=COLORS['card_bg']).pack(anchor='w', pady=2)
        
        # Dataset Source Indicator
        source_frame = tk.Frame(main, bg=COLORS['bg_secondary'], relief='solid', bd=1)
        source_frame.pack(pady=20, fill='x', padx=100)
        
        tk.Label(source_frame, text="📊 DATA SOURCE:",
                font=('Arial', 12, 'bold'),
                bg=COLORS['bg_secondary']).pack(pady=5)
        
        self.source_label = tk.Label(source_frame, 
                                    text="Not loaded",
                                    font=('Arial', 14, 'bold'),
                                    bg=COLORS['bg_secondary'],
                                    fg=COLORS['text_dark'])
        self.source_label.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(main, text="⏳ Ready to load dataset...",
                                    font=('Arial', 12),
                                    bg=COLORS['bg_primary'],
                                    fg=COLORS['text_dark'])
        self.status_label.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(main, bg=COLORS['bg_primary'])
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text='📊 Load Dataset',
                 command=self.load_file,
                 bg=COLORS['accent_green'],
                 font=('Arial', 12, 'bold'),
                 padx=30, pady=10,
                 cursor='hand2').pack(side='left', padx=10)
    
    # ================= DATASET PAGE =================
    def create_dataset_page(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='📊 Dataset Info')
        
        # Main container
        main = tk.Frame(frame, bg=COLORS['bg_primary'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(main, text='Dataset Information',
                font=('Arial', 24, 'bold'),
                bg=COLORS['bg_primary'],
                fg=COLORS['text_dark']).pack(anchor='w', pady=(0, 20))
        
        # Source info card
        source_card = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        source_card.pack(fill='x', pady=10, ipady=10)
        
        tk.Label(source_card, text="📁 DATA SOURCE DETAILS",
                font=('Arial', 14, 'bold'),
                bg=COLORS['card_bg'],
                fg=COLORS['accent_green_dark']).pack(anchor='w', padx=10, pady=5)
        
        self.source_details = tk.Label(source_card, 
                                      text="No dataset loaded",
                                      font=('Arial', 11),
                                      bg=COLORS['card_bg'],
                                      justify='left',
                                      anchor='w')
        self.source_details.pack(anchor='w', padx=20, pady=5)
        
        # Path display
        path_frame = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        path_frame.pack(fill='x', pady=10, ipady=10)
        
        tk.Label(path_frame, text='📍 Current Path:',
                font=('Arial', 11, 'bold'),
                bg=COLORS['card_bg']).pack(anchor='w', padx=10, pady=5)
        
        self.path_var = tk.StringVar(value="No file loaded")
        path_entry = tk.Entry(path_frame, textvariable=self.path_var,
                            font=('Consolas', 10),
                            bg=COLORS['bg_secondary'],
                            width=80,
                            state='readonly')
        path_entry.pack(fill='x', padx=10, pady=5)
        
        # Info display
        self.info_text = tk.Text(main, height=20,
                                font=('Courier', 10),
                                bg=COLORS['bg_secondary'])
        self.info_text.pack(fill='both', expand=True, pady=10)
    
    # ================= PATIENTS PAGE =================
    def create_patients_page(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='👥 Patients')
        
        # Main container
        main = tk.Frame(frame, bg=COLORS['bg_primary'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title and selection indicator
        title_frame = tk.Frame(main, bg=COLORS['bg_primary'])
        title_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(title_frame, text='Patient Records',
                font=('Arial', 24, 'bold'),
                bg=COLORS['bg_primary'],
                fg=COLORS['text_dark']).pack(side='left')
        
        self.selected_indicator = tk.Label(title_frame, 
                                         text="👤 No patient selected",
                                         font=('Arial', 12, 'bold'),
                                         bg=COLORS['bg_secondary'],
                                         fg=COLORS['text_dark'],
                                         padx=15, pady=5,
                                         relief='solid', bd=1)
        self.selected_indicator.pack(side='right')
        
        # Create treeview with scrollbars
        tree_frame = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        v_scroll = tk.Scrollbar(tree_frame)
        v_scroll.pack(side='right', fill='y')
        
        h_scroll = tk.Scrollbar(tree_frame, orient='horizontal')
        h_scroll.pack(side='bottom', fill='x')
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, 
                                 yscrollcommand=v_scroll.set,
                                 xscrollcommand=h_scroll.set,
                                 height=15,
                                 selectmode='browse')
        
        v_scroll.config(command=self.tree.yview)
        h_scroll.config(command=self.tree.xview)
        
        self.tree.pack(side='left', fill='both', expand=True)
        
        # Bind selection events
        self.tree.bind('<<TreeviewSelect>>', self.on_patient_select)
        self.tree.bind('<Double-Button-1>', self.on_patient_double_click)
        
        # Patient details panel
        details_frame = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        details_frame.pack(fill='x', pady=10, ipady=10)
        
        tk.Label(details_frame, text='📋 Selected Patient Details:',
                font=('Arial', 14, 'bold'),
                bg=COLORS['card_bg']).pack(anchor='w', padx=10)
        
        self.details_label = tk.Label(details_frame, 
                                     text="Click on a patient to view details",
                                     font=('Arial', 11),
                                     bg=COLORS['card_bg'],
                                     fg=COLORS['text_dark'],
                                     justify='left')
        self.details_label.pack(anchor='w', padx=20, pady=5)
        
        # Action buttons
        action_frame = tk.Frame(main, bg=COLORS['bg_primary'])
        action_frame.pack(pady=10)
        
        self.analyze_btn = tk.Button(action_frame,
                                    text='📊 Analyze Selected Patient',
                                    command=self.analyze_selected_patient,
                                    font=('Arial', 11, 'bold'),
                                    bg=COLORS['accent_green'],
                                    fg=COLORS['text_dark'],
                                    padx=20, pady=8,
                                    state='disabled',
                                    cursor='hand2')
        self.analyze_btn.pack(side='left', padx=5)
        
        self.report_btn = tk.Button(action_frame,
                                   text='📋 View Report',
                                   command=self.view_patient_report,
                                   font=('Arial', 11),
                                   bg=COLORS['accent_pink'],
                                   fg=COLORS['text_dark'],
                                   padx=20, pady=8,
                                   state='disabled',
                                   cursor='hand2')
        self.report_btn.pack(side='left', padx=5)
        
        # Patient count
        self.patient_count_label = tk.Label(main, 
                                           text="No patients loaded",
                                           font=('Arial', 10),
                                           bg=COLORS['bg_primary'],
                                           fg=COLORS['text_dark'])
        self.patient_count_label.pack(pady=5)
    
    # ================= ANALYSIS PAGE =================
    def create_analysis_page(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='📈 Analysis')
        
        # Main container
        main = tk.Frame(frame, bg=COLORS['bg_primary'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Patient info bar
        info_bar = tk.Frame(main, bg=COLORS['accent_blue'], relief='solid', bd=1)
        info_bar.pack(fill='x', pady=(0, 10), ipady=5)
        
        tk.Label(info_bar, text='👤 Currently Analyzing:',
                font=('Arial', 12, 'bold'),
                bg=COLORS['accent_blue'],
                fg=COLORS['text_dark']).pack(side='left', padx=10)
        
        self.analysis_patient_label = tk.Label(info_bar,
                                             text="No patient selected",
                                             font=('Arial', 12, 'bold'),
                                             bg=COLORS['accent_blue'],
                                             fg=COLORS['text_dark'])
        self.analysis_patient_label.pack(side='left', padx=5)
        
        # Source info
        self.analysis_source = tk.Label(info_bar,
                                       text="",
                                       font=('Arial', 10),
                                       bg=COLORS['accent_blue'],
                                       fg=COLORS['text_dark'])
        self.analysis_source.pack(side='right', padx=10)
        
        # Control panel - Simplified with only Savitzky-Golay filter
        control = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        control.pack(fill='x', pady=10, ipady=10)
        
        tk.Label(control, text='⚙️ Filter Settings:',
                font=('Arial', 11, 'bold'),
                bg=COLORS['card_bg']).pack(side='left', padx=10)
        
        # Window parameter
        tk.Label(control, text='Window:',
                bg=COLORS['card_bg']).pack(side='left', padx=5)
        self.window_var = tk.StringVar(value='21')
        tk.Spinbox(control, from_=3, to=51, increment=2,
                  textvariable=self.window_var,
                  width=8, bg=COLORS['bg_secondary']).pack(side='left')
        
        # Order parameter
        tk.Label(control, text='Order:',
                bg=COLORS['card_bg']).pack(side='left', padx=5)
        self.order_var = tk.StringVar(value='3')
        tk.Spinbox(control, from_=2, to=5,
                  textvariable=self.order_var,
                  width=8, bg=COLORS['bg_secondary']).pack(side='left')
        
        # Filter indicator (shows we're using Savitzky-Golay)
        tk.Label(control, text='🔍 Filter: Savitzky-Golay',
                font=('Arial', 10, 'italic'),
                bg=COLORS['card_bg'],
                fg=COLORS['accent_green_dark']).pack(side='left', padx=15)
        
        # Update button
        tk.Button(control, text='Apply Filter & Update Analysis',
                 command=self.update_analysis,
                 bg=COLORS['accent_green'],
                 padx=15, pady=2).pack(side='left', padx=10)
        
        # Create figure for plotting
        self.fig = Figure(figsize=(12, 6), facecolor=COLORS['bg_secondary'])
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        
        self.ax1.set_facecolor(COLORS['bg_secondary'])
        self.ax2.set_facecolor(COLORS['bg_secondary'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, main)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, main)
        toolbar.update()
        
        # Metrics panel
        metrics = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        metrics.pack(fill='x', pady=10, ipady=10)
        
        # Metrics labels
        metrics_frame = tk.Frame(metrics, bg=COLORS['card_bg'])
        metrics_frame.pack(expand=True)
        
        self.metrics = {}
        metric_items = [
            ('❤️ Heart Rate', '-- BPM'),
            ('📊 GSR Mean', '-- µS'),
            ('📈 Stress Level', '--%'),
            ('🔍 Peaks', '--'),
            ('😊 Status', '--')
        ]
        
        for label, default in metric_items:
            frame = tk.Frame(metrics_frame, bg=COLORS['card_bg'])
            frame.pack(side='left', padx=20)
            
            tk.Label(frame, text=label,
                    font=('Arial', 11, 'bold'),
                    bg=COLORS['card_bg'],
                    fg=COLORS['text_dark']).pack()
            
            self.metrics[label] = tk.Label(frame, text=default,
                                          font=('Arial', 14),
                                          bg=COLORS['card_bg'],
                                          fg=COLORS['accent_green_dark'])
            self.metrics[label].pack()
    
    # ================= REPORT PAGE =================
    def create_report_page(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='📋 Report')
        
        # Main container
        main = tk.Frame(frame, bg=COLORS['bg_primary'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header with team info
        header = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        header.pack(fill='x', pady=(0, 20), ipady=10)
        
        # Team info in header
        info_frame = tk.Frame(header, bg=COLORS['card_bg'])
        info_frame.pack(padx=20)
        
        team_text = "Syeda Memoona Urooj (23-CE-006) | Areeba Khalique (23-CE-016) | Supervisor: Mam Hareem"
        tk.Label(info_frame, text=team_text,
                font=('Arial', 11, 'bold'),
                bg=COLORS['card_bg'],
                fg=COLORS['text_dark']).pack()
        
        tk.Label(info_frame, text="Department of Computer Engineering",
                font=('Arial', 10),
                bg=COLORS['card_bg'],
                fg=COLORS['accent_green_dark']).pack()
        
        # Report title
        tk.Label(main, text='Stress Analysis Report',
                font=('Arial', 24, 'bold'),
                bg=COLORS['bg_primary'],
                fg=COLORS['text_dark']).pack(anchor='w')
        
        # Report text area
        text_frame = tk.Frame(main, bg=COLORS['card_bg'], relief='solid', bd=1)
        text_frame.pack(fill='both', expand=True, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.report_text = tk.Text(text_frame,
                                  font=('Courier', 11),
                                  bg=COLORS['bg_secondary'],
                                  wrap='word',
                                  padx=15, pady=15,
                                  yscrollcommand=scrollbar.set)
        self.report_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.report_text.yview)
        
        # Buttons
        btn_frame = tk.Frame(main, bg=COLORS['bg_primary'])
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text='📊 Generate Report',
                 command=self.generate_report,
                 font=('Arial', 11, 'bold'),
                 bg=COLORS['accent_green'],
                 fg=COLORS['text_dark'],
                 padx=20, pady=8,
                 cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text='💾 Save Report',
                 command=self.save_report,
                 font=('Arial', 11),
                 bg=COLORS['accent_pink'],
                 fg=COLORS['text_dark'],
                 padx=20, pady=8,
                 cursor='hand2').pack(side='left', padx=5)
    
    # ================= FILE LOADING =================
    def load_file(self):
        """Load dataset file with string handling"""
        file = filedialog.askopenfilename(
            title="Select Stress Dataset",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not file:
            return
        
        try:
            self.status_label.config(text="⏳ Loading dataset...", fg=COLORS['text_dark'])
            self.root.update()
            
            # Try to load based on extension
            if file.endswith('.csv'):
                self.dataset_df = pd.read_csv(file, dtype=str)  # Load all as string first
            else:
                self.dataset_df = pd.read_excel(file, dtype=str)  # Load all as string first
            
            self.dataset_path = file
            self.path_var.set(file)
            
            # Update source information
            self.update_source_info(file)
            
            # Display dataset info
            self.display_dataset_info()
            
            # Load patients into treeview
            self.load_patients()
            
            self.status_label.config(text=f"✅ Loaded: {self.dataset_df.shape[0]} rows", 
                                    fg=COLORS['accent_green_dark'])
            
            messagebox.showinfo("Success", f"Dataset loaded successfully!\n{self.dataset_df.shape[0]} rows")
            
        except Exception as e:
            self.status_label.config(text=f"❌ Error: {str(e)[:50]}", fg='red')
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def update_source_info(self, path):
        """Update source information displays"""
        file_size = os.path.getsize(path) / 1024  # KB
        modified_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        
        source_text = f"""
📁 File: {os.path.basename(path)}
📂 Location: {os.path.dirname(path)}
📊 Size: {file_size:.2f} KB
🕒 Modified: {modified_time}
        """
        
        self.source_label.config(text=f"📊 {os.path.basename(path)}")
        self.source_details.config(text=source_text)
    
    def display_dataset_info(self):
        """Display dataset information"""
        if self.dataset_df is None:
            return
        
        info = f"""
╔════════════════════════════════════════════════════════════╗
║                    DATASET INFORMATION                     ║
╚════════════════════════════════════════════════════════════╝

📊 Dataset Shape: {self.dataset_df.shape[0]} rows × {self.dataset_df.shape[1]} columns

📋 Column Names:
"""
        for i, col in enumerate(self.dataset_df.columns, 1):
            info += f"   {i}. {col}\n"
        
        info += f"\n📈 First 5 rows:\n{self.dataset_df.head().to_string()}"
        
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info)
    
    # ================= PATIENT LOADING =================
    def load_patients(self):
        """Load patients into treeview"""
        if self.dataset_df is None:
            return
        
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure columns based on dataframe
        columns = ['ID'] + list(self.dataset_df.columns[:4]) + ['Stress %', 'Status']
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        
        self.patient_details.clear()
        
        # Add rows
        for idx, row in self.dataset_df.head(100).iterrows():
            # Get values as strings
            values = []
            for i, col in enumerate(self.dataset_df.columns[:4]):
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                else:
                    values.append(str(val)[:15])
            
            while len(values) < 4:
                values.append("--")
            
            # Calculate stress based on text content
            stress = 30 + (idx * 2) % 70
            
            # Check for stress-related keywords
            text_content = ' '.join(values).lower()
            if any(word in text_content for word in ['stress', 'anxiety', 'ptsd']):
                stress = 85
            elif any(word in text_content for word in ['calm', 'relax']):
                stress = 20
            
            # Try to use numeric values if available
            try:
                if values[0] not in ['--', 'N/A']:
                    # Try to extract number from string
                    num_str = ''.join(c for c in values[0] if c.isdigit() or c == '.' or c == '-')
                    if num_str:
                        val1 = float(num_str)
                        stress = (val1 * 10) % 100
            except:
                pass
            
            # Determine status
            if stress < 30:
                status = "😊 Relaxed"
                bg_color = COLORS['stress_low']
            elif stress < 60:
                status = "😐 Mild Stress"
                bg_color = COLORS['stress_medium']
            else:
                status = "😰 High Stress"
                bg_color = COLORS['stress_high']
            
            patient_id = f"P{idx+1:04d}"
            
            # Store details with safe float conversion
            ppg_val = safe_float_conversion(values[0], 0.5)
            gsr_val = safe_float_conversion(values[1], 2.0)
            
            self.patient_details[patient_id] = {
                'index': idx,
                'row': row.to_dict(),
                'values': values,
                'stress': stress,
                'status': status,
                'ppg': ppg_val,
                'gsr': gsr_val,
                'raw_text': text_content
            }
            
            # Insert into tree
            item_id = self.tree.insert('', 'end', values=[
                patient_id, values[0], values[1], values[2], values[3],
                f"{stress:.1f}%", status
            ])
            
            # Apply tag for coloring
            self.tree.item(item_id, tags=(status,))
        
        # Configure tag colors
        self.tree.tag_configure("😊 Relaxed", background=COLORS['stress_low'])
        self.tree.tag_configure("😐 Mild Stress", background=COLORS['stress_medium'])
        self.tree.tag_configure("😰 High Stress", background=COLORS['stress_high'])
        
        self.patient_count_label.config(text=f"📊 Loaded {len(self.patient_details)} patients from {os.path.basename(self.dataset_path) if self.dataset_path else 'Unknown'}")
    
    # ================= PATIENT SELECTION =================
    def on_patient_select(self, event):
        """Handle patient selection"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        values = item['values']
        patient_id = values[0]
        
        self.selected_patient = values
        self.selected_patient_id = patient_id
        
        # Update selection indicator
        self.selected_indicator.config(
            text=f"👤 Selected: {patient_id} - {values[6]}",
            bg=COLORS['selected_bg']
        )
        
        # Enable buttons
        self.analyze_btn.config(state='normal')
        self.report_btn.config(state='normal')
        
        # Update details
        details = f"Patient ID: {patient_id}\n"
        details += f"Value 1: {values[1]}\n"
        details += f"Value 2: {values[2]}\n"
        details += f"Value 3: {values[3]}\n"
        details += f"Value 4: {values[4]}\n"
        details += f"Stress Level: {values[5]}\n"
        details += f"Status: {values[6]}\n"
        details += f"Data Source: {os.path.basename(self.dataset_path) if self.dataset_path else 'Not loaded'}"
        
        self.details_label.config(text=details)
    
    def on_patient_double_click(self, event):
        """Double click to analyze"""
        self.analyze_selected_patient()
    
    def analyze_selected_patient(self):
        """Analyze selected patient"""
        if not self.selected_patient:
            messagebox.showwarning("Warning", "Select a patient first!")
            return
        
        self.notebook.select(3)  # Switch to analysis page
        patient_id = self.selected_patient[0]
        
        # Update analysis page
        self.analysis_patient_label.config(
            text=f"{patient_id} - {self.selected_patient[6]}"
        )
        self.analysis_source.config(
            text=f"Source: {os.path.basename(self.dataset_path) if self.dataset_path else 'Unknown'}"
        )
        
        # Get patient details
        details = self.patient_details.get(patient_id, {})
        stress = details.get('stress', 50)
        ppg_val = details.get('ppg', 0.5)
        gsr_val = details.get('gsr', 2.0)
        
        # Create synthetic time and signals based on stress level
        self.time_data = np.linspace(0, 10, 1000)
        
        if stress > 60:
            # High stress - higher frequency, more variability
            self.ppg_data = 0.8 * np.sin(2 * np.pi * 1.5 * self.time_data) + \
                           0.4 * np.sin(2 * np.pi * 3.0 * self.time_data) + \
                           0.2 * np.random.randn(len(self.time_data))
            self.gsr_data = 5.0 + 1.0 * np.sin(2 * np.pi * 0.2 * self.time_data) + \
                           0.3 * np.random.randn(len(self.time_data))
        elif stress > 30:
            # Mild stress
            self.ppg_data = 0.6 * np.sin(2 * np.pi * 1.3 * self.time_data) + \
                           0.3 * np.sin(2 * np.pi * 2.6 * self.time_data) + \
                           0.15 * np.random.randn(len(self.time_data))
            self.gsr_data = 3.5 + 0.5 * np.sin(2 * np.pi * 0.1 * self.time_data) + \
                           0.2 * np.random.randn(len(self.time_data))
        else:
            # Relaxed
            self.ppg_data = 0.4 * np.sin(2 * np.pi * 1.1 * self.time_data) + \
                           0.2 * np.sin(2 * np.pi * 2.2 * self.time_data) + \
                           0.1 * np.random.randn(len(self.time_data))
            self.gsr_data = 2.0 + 0.3 * np.sin(2 * np.pi * 0.05 * self.time_data) + \
                           0.1 * np.random.randn(len(self.time_data))
        
        # Scale signals based on patient's actual values
        self.ppg_data = self.ppg_data * (ppg_val / np.mean(np.abs(self.ppg_data)))
        self.gsr_data = self.gsr_data * (gsr_val / np.mean(self.gsr_data))
        
        # Add stress event in middle if high stress
        if stress > 60:
            stress_idx = len(self.time_data) // 2
            self.ppg_data[stress_idx:stress_idx+200] *= 1.3
            self.gsr_data[stress_idx:stress_idx+200] += 0.8
        
        # Reset filtered signals
        self.filtered_ppg = None
        self.filtered_gsr = None
        
        # Plot initial signals
        self.plot_initial_signals()
        self.reset_metrics()
        
        messagebox.showinfo("Patient Selected", 
                           f"Now analyzing: {patient_id}\n"
                           f"Status: {self.selected_patient[6]}")
    
    def plot_initial_signals(self):
        """Plot initial signals"""
        if self.ppg_data is not None:
            self.ax1.clear()
            self.ax2.clear()
            
            self.ax1.plot(self.time_data, self.ppg_data, color=COLORS['accent_pink'], linewidth=1.5)
            self.ax1.set_title(f'PPG Signal - Patient {self.selected_patient[0]}')
            self.ax1.set_xlabel('Time (seconds)')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.grid(True, alpha=0.3)
            
            self.ax2.plot(self.time_data, self.gsr_data, color=COLORS['accent_green'], linewidth=1.5)
            self.ax2.set_title('GSR Signal')
            self.ax2.set_xlabel('Time (seconds)')
            self.ax2.set_ylabel('Conductance (µS)')
            self.ax2.grid(True, alpha=0.3)
            
            self.canvas.draw()
    
    def reset_metrics(self):
        """Reset metrics to default"""
        for key in self.metrics:
            if key == '❤️ Heart Rate':
                self.metrics[key].config(text='-- BPM')
            elif key == '📊 GSR Mean':
                self.metrics[key].config(text='-- µS')
            elif key == '📈 Stress Level':
                self.metrics[key].config(text='--%')
            elif key == '🔍 Peaks':
                self.metrics[key].config(text='--')
            elif key == '😊 Status':
                self.metrics[key].config(text='--')
        
        self.filtered_ppg = None
        self.filtered_gsr = None
    
    # ================= FILTERING =================
    def update_analysis(self):
        """Update analysis with Savitzky-Golay filter"""
        if self.ppg_data is None:
            messagebox.showwarning("Warning", "Please analyze a patient first!")
            return
        
        try:
            # Get filter parameters
            window = int(self.window_var.get())
            order = int(self.order_var.get())
            
            # Apply Savitzky-Golay filter
            self.filtered_ppg = apply_savgol_filter(self.ppg_data, window, order)
            self.filtered_gsr = apply_savgol_filter(self.gsr_data, window, order)
            
            # Detect peaks
            peaks, properties = find_peaks(self.filtered_ppg, distance=50, prominence=0.1)
            
            # Calculate metrics
            if len(peaks) > 1:
                hr = calculate_heart_rate(peaks, self.time_data)
                gsr_mean = np.mean(self.filtered_gsr)
                stress_status = detect_stress(self.filtered_gsr)
                stress_level = calculate_stress_level(hr, gsr_mean)
                
                # Update metrics
                self.metrics['❤️ Heart Rate'].config(text=f"{hr:.1f} BPM")
                self.metrics['📊 GSR Mean'].config(text=f"{gsr_mean:.3f} µS")
                self.metrics['📈 Stress Level'].config(text=f"{stress_level:.1f}%")
                self.metrics['🔍 Peaks'].config(text=str(len(peaks)))
                self.metrics['😊 Status'].config(text=stress_status)
                
                # Update plot with peaks
                self.ax1.clear()
                self.ax2.clear()
                
                # Plot PPG with peaks
                self.ax1.plot(self.time_data, self.ppg_data, alpha=0.3, color='gray', label='Raw PPG')
                self.ax1.plot(self.time_data, self.filtered_ppg, color=COLORS['accent_pink'], 
                            linewidth=2, label='Filtered PPG (Savitzky-Golay)')
                self.ax1.plot(self.time_data[peaks], self.filtered_ppg[peaks], 'ro', 
                            markersize=6, label='Peaks')
                self.ax1.set_title(f'PPG Signal - HR: {hr:.1f} BPM')
                self.ax1.set_xlabel('Time (seconds)')
                self.ax1.set_ylabel('Amplitude')
                self.ax1.legend(loc='upper right', fontsize=8)
                self.ax1.grid(True, alpha=0.3)
                
                # Plot GSR
                self.ax2.plot(self.time_data, self.gsr_data, alpha=0.3, color='gray', label='Raw GSR')
                self.ax2.plot(self.time_data, self.filtered_gsr, color=COLORS['accent_green'], 
                            linewidth=2, label='Filtered GSR (Savitzky-Golay)')
                self.ax2.set_title(f'GSR Signal - Status: {stress_status}')
                self.ax2.set_xlabel('Time (seconds)')
                self.ax2.set_ylabel('Conductance (µS)')
                self.ax2.legend(loc='upper right', fontsize=8)
                self.ax2.grid(True, alpha=0.3)
                
                self.canvas.draw()
                
                messagebox.showinfo("Analysis Complete", 
                                   f"Analysis complete!\n\n"
                                   f"Heart Rate: {hr:.1f} BPM\n"
                                   f"Stress Level: {stress_level:.1f}%\n"
                                   f"Status: {stress_status}")
            else:
                messagebox.showwarning("Warning", "Not enough peaks detected! Try adjusting filter parameters.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    # ================= REPORT FUNCTIONS =================
    def view_patient_report(self):
        """View report for selected patient"""
        if self.selected_patient:
            self.notebook.select(4)
            self.generate_report()
        else:
            messagebox.showwarning("Warning", "Please select a patient first!")
    
    def generate_report(self):
        """Generate report for selected patient"""
        if not self.selected_patient:
            messagebox.showwarning("Warning", "Please select a patient first!")
            return
        
        patient_id = self.selected_patient[0]
        details = self.patient_details.get(patient_id, {})
        
        # Get current analysis results
        hr_text = self.metrics['❤️ Heart Rate'].cget('text')
        gsr_text = self.metrics['📊 GSR Mean'].cget('text')
        stress_text = self.metrics['📈 Stress Level'].cget('text')
        peaks_text = self.metrics['🔍 Peaks'].cget('text')
        status_text = self.metrics['😊 Status'].cget('text')
        
        report = f"""
{'='*70}
              STRESS ANALYSIS REPORT
{'='*70}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {os.path.basename(self.dataset_path) if self.dataset_path else 'Unknown'}

{'='*70}
TEAM INFORMATION
{'='*70}
Department: Computer Engineering
Supervisor: Mam Hareem

Team Members:
  • Syeda Memoona Urooj (23-CE-006)
  • Areeba Khalique (23-CE-016)

{'='*70}
PATIENT INFORMATION
{'='*70}
Patient ID: {patient_id}
Value 1: {self.selected_patient[1]}
Value 2: {self.selected_patient[2]}
Value 3: {self.selected_patient[3]}
Value 4: {self.selected_patient[4]}
Base Stress Level: {self.selected_patient[5]}
Initial Status: {self.selected_patient[6]}

{'='*70}
ANALYSIS RESULTS
{'='*70}
Heart Rate: {hr_text}
GSR Mean: {gsr_text}
Calculated Stress: {stress_text}
Peaks Detected: {peaks_text}
Final Status: {status_text}

{'='*70}
FILTER INFORMATION
{'='*70}
Filter Type: Savitzky-Golay
Window Size: {self.window_var.get()}
Polynomial Order: {self.order_var.get()}

{'='*70}
CLINICAL INTERPRETATION
{'='*70}
"""
        
        # Add interpretation based on final status
        if "Stressed" in status_text:
            report += """
🔴 HIGH STRESS DETECTED
• Patient shows significantly elevated stress indicators
• Heart rate and GSR levels are above normal range
• Immediate intervention recommended
• Practice deep breathing exercises
• Consider stress management consultation
"""
        elif "Relaxed" in status_text:
            report += """
✅ RELAXED STATE
• Patient is in a calm and relaxed state
• Normal physiological parameters
• No significant stress indicators
• Continue current routine
• Regular monitoring recommended
"""
        else:
            report += """
⚠️ INCONCLUSIVE
• Unable to definitively determine stress state
• Consider longer monitoring period
• Check signal quality
• May need additional data
"""
        
        report += f"""
{'='*70}
DATASET STATISTICS
{'='*70}
Total Patients in Dataset: {len(self.patient_details)}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
                  END OF REPORT
{'='*70}
"""
        
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', report)
    
    def save_report(self):
        """Save report to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.report_text.get('1.0', tk.END))
            messagebox.showinfo("Success", f"Report saved to {file_path}")

# ================= RUN =================
def main():
    root = tk.Tk()
    app = StressMonitoringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()