from psychopy import visual, core, event

class ConcentrationBarGUI:
    def __init__(self, baseline_mu, baseline_sigma, threshold_z=2.0):
        # Window Setup: Larger size to fill screen (1600x900)
        self.win = visual.Window(size=(1600, 900), color='#050914', units='pix', fullscr=False)
        self.threshold_z = threshold_z
        self.min_z = -2.0
        self.max_z = 4.0
        
        # --- 1. TOP BAR (Concentration) ---
        # Track
        self.conc_track = visual.Rect(self.win, width=1000, height=6, pos=(-50, 380), 
                                      fillColor='#1A2035', lineColor=None)
        # Fill (Dynamic)
        self.conc_fill = visual.Rect(self.win, width=0, height=6, pos=(-550, 380), 
                                     fillColor='#A855F7', lineColor=None)
        
        # Percentage text
        self.pct_text = visual.TextStim(self.win, text='0%', pos=(470, 395), 
                                        color='#FFFFFF', height=16, bold=True, font='Arial')
        # Status text
        self.status_text = visual.TextStim(self.win, text='IDLE', pos=(650, 380), 
                                           color='#A855F7', height=18, bold=True, font='Arial')

        # --- 2. P300 GRID (SHIFTED LEFT) ---
        self.grid_items = []
        chars = ['A','B','C','D','E','F', 'G','H','I','J','K','L', 
                 'M','N','O','P','Q','R', 'S','T','U','V','W','X', 
                 'Y','Z','0','1','2','3']
        
        # Dimensions
        grid_w, grid_h = 160, 110
        gap = 15
        # SHIFTED LEFT: Start at -750 (was -650) to make room on the right
        start_x = -650
        start_y = 200
        
        for i, char in enumerate(chars):
            row = i // 6
            col = i % 6
            x = start_x + col * (grid_w + gap)
            y = start_y - row * (grid_h + gap)
            
            btn = visual.Rect(self.win, width=grid_w, height=grid_h, pos=(x, y),
                              fillColor='#0F172A', lineColor='#1E293B', lineWidth=2)
            txt = visual.TextStim(self.win, text=char, pos=(x, y), 
                                  color='#E2E8F0', height=36, bold=True, font='Arial')
            self.grid_items.append((btn, txt))

        # --- 3. SSVEP SIDE PANELS (SHIFTED RIGHT & CENTERED) ---
        self.ssvep_blocks = []
        # Dimensions
        block_w, block_h = 280, 220
        
        # SHIFTED RIGHT: Start at 550 (was 350) to sit in the empty space
        # This creates a ~100px gap between the Keyboard and SSVEP blocks
        start_x_right = 550  
        
        # SHIFTED DOWN slightly to center vertically relative to the keyboard
        start_y_right = 210
        
        labels = ['Word 1', 'Word 2', 'Word 3']
        freqs = ['8.0 Hz', '10.0 Hz', '12.0 Hz']
        
        for i in range(3):
            y = start_y_right - i * (block_h + 15)
            
            box = visual.Rect(self.win, width=block_w, height=block_h, pos=(start_x_right, y),
                              fillColor='#0A0F1E', lineColor='#1E293B', lineWidth=2)
            
            w_txt = visual.TextStim(self.win, text=labels[i], pos=(start_x_right, y + 50), 
                                    color='#94A3B8', height=26, font='Arial')
            f_txt = visual.TextStim(self.win, text=freqs[i], pos=(start_x_right, y - 10), 
                                    color='#475569', height=18, font='Arial')
            
            self.ssvep_blocks.append((box, w_txt, f_txt))

    def update(self, ratio, z_score):
        # 1. Update Concentration Bar
        fill_pct = max(0.0, min(1.0, (z_score - self.min_z) / (self.max_z - self.min_z)))
        new_width = fill_pct * 1000
        
        self.conc_fill.width = new_width
        self.conc_fill.pos = (-550 + new_width/2, 380)
        
        # Color logic
        if z_score >= self.threshold_z:
            self.conc_fill.fillColor = '#00E5FF'
            self.status_text.text, self.status_text.color = 'HIGH FOCUS', '#00E5FF'
        else:
            self.conc_fill.fillColor = '#A855F7'
            self.status_text.text, self.status_text.color = 'IDLE', '#A855F7'
            
        self.pct_text.text = f'{int(fill_pct * 100)}%'
        self.pct_text.pos = (-550 + new_width + 10, 395)

        # 2. Draw Everything
        # Grid
        for btn, txt in self.grid_items:
            btn.draw()
            txt.draw()
            
        # SSVEP
        for box, w, f in self.ssvep_blocks:
            box.draw()
            w.draw()
            f.draw()
            
        # Top UI
        self.conc_track.draw()
        self.conc_fill.draw()
        self.pct_text.draw()
        self.status_text.draw()
        
        self.win.flip()
        
    def close(self):
        self.win.close()