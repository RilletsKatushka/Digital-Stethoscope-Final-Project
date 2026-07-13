import os
import sys
import time
import csv
import wave
import sqlite3
import subprocess
from threading import Thread, Lock
from collections import deque

import wx
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.ticker import ScalarFormatter
import matplotlib.animation as animation

FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def gender_label_to_code(label: str) -> str:
    s = (label or "").strip().lower()
    if s in ("laki-laki", "laki laki", "l", "male", "pria"): return "L"
    if s in ("perempuan", "p", "female", "wanita"): return "P"
    return ""

def gender_code_to_label(code: str) -> str:
    c = (code or "").strip().upper()
    if c == "L": return "Laki-laki"
    if c == "P": return "Perempuan"
    return ""

def app_dir():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

DB_PATH = os.path.join(app_dir(), 'stethoscope.db')

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT,
            name TEXT,
            birthdate TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            exam_date TEXT,
            wav_path TEXT,
            png_path TEXT,
            video_path TEXT,
            duration REAL,
            peak_amp REAL,
            dominant_freq REAL,
            bpm REAL,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    """)
    # backfill gender column if missing
    try:
        cur.execute("PRAGMA table_info(patients)")
        cols = [r[1].lower() for r in cur.fetchall()]
        if 'gender' not in cols:
            cur.execute("ALTER TABLE patients ADD COLUMN gender TEXT DEFAULT ''")
    except Exception:
        pass
    # normalize gender values
    try:
        cur.execute("""
            UPDATE patients
            SET gender = CASE
                WHEN lower(trim(gender)) IN ('l','laki-laki','laki laki','male','pria') THEN 'L'
                WHEN lower(trim(gender)) IN ('p','perempuan','female','wanita') THEN 'P'
                ELSE COALESCE(NULLIF(gender,''), '')
            END
            WHERE gender IS NOT NULL
        """)
    except Exception:
        pass
    # ensure bpm column exists for older DBs
    try:
        cur.execute("PRAGMA table_info(recordings)")
        rcols = [r[1].lower() for r in cur.fetchall()]
        if 'bpm' not in rcols:
            cur.execute("ALTER TABLE recordings ADD COLUMN bpm REAL")
    except Exception:
        pass
    con.commit(); con.close()


def upsert_patient(patient_code: str, name: str, birthdate: str, gender_code: str):
    gender_code = gender_label_to_code(gender_code) if len(gender_code) > 1 else (gender_code or "")
    if gender_code not in ("L","P",""): gender_code = ""
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id FROM patients WHERE patient_code = ?", (patient_code.strip(),))
    row = cur.fetchone()
    if row:
        pid = row[0]
        cur.execute("UPDATE patients SET name=?, birthdate=?, gender=? WHERE id=?",
                    (name.strip(), birthdate.strip(), gender_code, pid))
    else:
        cur.execute("INSERT INTO patients(patient_code,name,birthdate,gender) VALUES(?,?,?,?)",
                    (patient_code.strip(), name.strip(), birthdate.strip(), gender_code))
        pid = cur.lastrowid
    con.commit(); con.close(); return pid


def insert_recording(patient_id:int, exam_date:str, wav_path:str, png_path:str, video_path:str,
                     duration:float, peak_amp:float, dominant_freq:float, bpm:float):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""
        INSERT INTO recordings(patient_id,exam_date,wav_path,png_path,video_path,duration,peak_amp,dominant_freq,bpm)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (patient_id, exam_date, wav_path, png_path, video_path, duration, peak_amp, dominant_freq, bpm))
    con.commit(); con.close()


def fetch_recent_recordings(limit=200):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""
        SELECT r.id, p.patient_code, p.name, p.gender, r.exam_date, r.wav_path, r.png_path, r.video_path,
               r.duration, r.peak_amp, r.dominant_freq, r.bpm
        FROM recordings r
        LEFT JOIN patients p ON p.id = r.patient_id
        ORDER BY r.id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall(); con.close(); return rows


def get_patient_by_code(patient_code:str):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id, name, birthdate, gender FROM patients WHERE patient_code = ? LIMIT 1",
                (patient_code.strip(),))
    row = cur.fetchone(); con.close(); return row


def get_recording_meta_by_wav(wav_path:str):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""
        SELECT p.patient_code, p.name, p.birthdate, p.gender, r.exam_date, r.bpm
        FROM recordings r LEFT JOIN patients p ON p.id = r.patient_id
        WHERE r.wav_path = ? LIMIT 1
    """, (wav_path,))
    row = cur.fetchone(); con.close(); return row


def get_recording_paths_by_id(rec_id:int):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT wav_path, png_path, video_path FROM recordings WHERE id=? LIMIT 1", (int(rec_id),))
    row = cur.fetchone(); con.close(); return None if row is None else tuple(row)


def delete_recording_by_id(rec_id:int):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("DELETE FROM recordings WHERE id=?", (int(rec_id),))
    con.commit(); con.close()


def export_db_to_csv(folder:str):
    os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT id, patient_code, name, birthdate, gender FROM patients ORDER BY id")
    rows = cur.fetchall()
    ppath = os.path.join(folder, "patients.csv")
    with open(ppath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","patient_code","name","birthdate","gender_code"]); w.writerows(rows)
    cur.execute("""
        SELECT id, patient_id, exam_date, wav_path, png_path, video_path, duration, peak_amp, dominant_freq, bpm
        FROM recordings ORDER BY id
    """)
    rows = cur.fetchall()
    rpath = os.path.join(folder, "recordings.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["id","patient_id","exam_date","wav_path","png_path","video_path","duration","peak_amp","dominant_freq","bpm"]
        ); w.writerows(rows)
    cur.execute("""
        SELECT r.id AS rec_id, p.patient_code, p.name, p.birthdate, p.gender,
               CASE UPPER(p.gender) WHEN 'L' THEN 'Laki-laki' WHEN 'P' THEN 'Perempuan' ELSE '' END AS gender_label,
               r.exam_date, r.wav_path, r.png_path, r.video_path, r.duration, r.peak_amp, r.dominant_freq, r.bpm
        FROM recordings r LEFT JOIN patients p ON p.id = r.patient_id
        ORDER BY r.id
    """)
    rows = cur.fetchall()
    jpath = os.path.join(folder, "recordings_joined.csv")
    with open(jpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["rec_id","patient_code","name","birthdate","gender_code","gender_label",
             "exam_date","wav_path","png_path","video_path","duration","peak_amp","dominant_freq","bpm"]
        )
        w.writerows(rows)
    con.close(); return ppath, rpath, jpath

def compute_peak_and_freq(audio_array: np.ndarray, rate: int):
    if audio_array.size == 0: return 0.0, 0.0
    peak = float(np.max(np.abs(audio_array)))
    window = np.hanning(audio_array.size); y = audio_array * window
    Y = np.fft.rfft(y); freqs = np.fft.rfftfreq(y.size, d=1.0/rate); mags = np.abs(Y)
    if mags.size > 1: mags[0] = 0
    band = (freqs >= 20) & (freqs <= 250)
    dom = float(freqs[band][np.argmax(mags[band])]) if np.any(band) else 0.0
    return peak, dom

def _fir_bandpass(sig, fs, low=20.0, high=200.0, numtaps=801):
    nyq = fs / 2.0
    lo = low / nyq
    hi = high / nyq
    M = int(numtaps)
    n = np.arange(M) - (M - 1) / 2.0

    h_hi = 2 * hi * np.sinc(2 * hi * n)
    h_lo = 2 * lo * np.sinc(2 * lo * n)
    h = (h_hi - h_lo) * np.hamming(M)
    h /= np.sum(h) + 1e-12
    return np.convolve(sig, h, mode='same')

def compute_bpm_and_beats(audio_array: np.ndarray, rate: int):
    if audio_array is None or audio_array.size == 0 or rate <= 0:
        return 0.0, []

    x = audio_array.astype(float)
    x -= np.mean(x)
    maxabs = np.max(np.abs(x)) + 1e-9
    x /= maxabs

    x = _fir_bandpass(x, rate, low=20.0, high=200.0, numtaps=801)

    env = np.abs(x)
    win = max(1, int(rate * 0.05))         # 50 ms
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode='same')

    thr = np.percentile(env, 85)

    refr = max(1, int(rate * 0.50))
    N = len(env)
    peaks = []
    i = 0
    while i < N:
        if env[i] > thr:
            j_end = min(N, i + refr)
            j = i + int(np.argmax(env[i:j_end]))
            peaks.append(j)
            i = j_end
        else:
            i += 1

    if len(peaks) < 2:
        return 0.0, []

    t = np.array(peaks) / float(rate)
    ibi = np.diff(t)

    mask = (ibi >= 0.40) & (ibi <= 1.50)
    ibi = ibi[mask]
    if ibi.size == 0:
        return 0.0, []

    bpm = 60.0 / float(np.median(ibi))
    return float(bpm), t.tolist()

class ExplorerPanel(wx.Panel):
    def __init__(self, parent, on_open_wav=None):
        super().__init__(parent); self.on_open_wav = on_open_wav
        splitter = wx.SplitterWindow(self)
        self.dir_tree = wx.GenericDirCtrl(splitter, style=wx.DIRCTRL_DIR_ONLY)
        self.file_list = wx.ListCtrl(splitter, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.file_list.InsertColumn(0, "Nama File", width=500)
        splitter.SplitVertically(self.dir_tree, self.file_list, sashPosition=280)
        sizer = wx.BoxSizer(wx.VERTICAL); sizer.Add(splitter, 1, wx.EXPAND); self.SetSizer(sizer)
        self.current_dir = os.path.expanduser('~'); self.dir_tree.SetPath(self.current_dir)
        self.Bind(wx.EVT_DIRCTRL_SELECTIONCHANGED, self.OnFolderChanged, self.dir_tree)
        self.file_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnFileActivated)
        self.file_list.Bind(wx.EVT_CONTEXT_MENU, self.OnRightClick)

    def OnFolderChanged(self, event):
        folder_path = self.dir_tree.GetPath(); self.UpdateFileList(folder_path)

    def UpdateFileList(self, folder_path):
        self.file_list.DeleteAllItems()
        if os.path.isdir(folder_path):
            for file_name in sorted(os.listdir(folder_path)):
                full_path = os.path.join(folder_path, file_name)
                if os.path.isfile(full_path):
                    self.file_list.InsertItem(self.file_list.GetItemCount(), file_name)
        self.current_dir = folder_path

    def OnFileActivated(self, event):
        idx = event.GetIndex(); file_name = self.file_list.GetItemText(idx)
        full_path = os.path.join(self.current_dir, file_name)
        if not os.path.isfile(full_path): return
        try:
            if file_name.lower().endswith('.wav') and callable(self.on_open_wav):
                self.on_open_wav(full_path); return
            if sys.platform.startswith('win'): os.startfile(full_path)
            elif sys.platform == 'darwin': subprocess.run(['open', full_path])
            else: subprocess.run(['xdg-open', full_path])
        except Exception as e:
            wx.MessageBox(f"Gagal membuka file:\n{e}", "Error", wx.ICON_ERROR)

    def OnRightClick(self, event):
        pos = self.ScreenToClient(event.GetPosition()); index, flags = self.file_list.HitTest(pos)
        if index == wx.NOT_FOUND: return
        self.file_list.Select(index)
        menu = wx.Menu(); rename_item = menu.Append(wx.ID_ANY, "Rename"); delete_item = menu.Append(wx.ID_ANY, "Delete")
        self.Bind(wx.EVT_MENU, lambda evt: self.RenameFile(index), rename_item)
        self.Bind(wx.EVT_MENU, lambda evt: self.DeleteFile(index), delete_item)
        self.PopupMenu(menu); menu.Destroy()

    def DeleteFile(self, index):
        file_name = self.file_list.GetItemText(index); full_path = os.path.join(self.current_dir, file_name)
        confirm = wx.MessageBox(f"Yakin ingin menghapus '{file_name}'?", "Konfirmasi Hapus",
                                wx.YES_NO | wx.ICON_QUESTION)
        if confirm != wx.YES: return
        try:
            os.remove(full_path); wx.MessageBox("File berhasil dihapus.", "Sukses", wx.OK | wx.ICON_INFORMATION)
            self.UpdateFileList(self.current_dir)
        except Exception as e:
            wx.MessageBox(f"Gagal menghapus file:\n{e}", "Error", wx.OK | wx.ICON_ERROR)

    def RenameFile(self, index):
        old_name = self.file_list.GetItemText(index); old_path = os.path.join(self.current_dir, old_name)
        dlg = wx.TextEntryDialog(self, "Masukkan nama baru:", "Rename File", old_name)
        if dlg.ShowModal() == wx.ID_OK:
            new_name = dlg.GetValue()
            if new_name and new_name != old_name:
                new_path = os.path.join(self.current_dir, new_name)
                try:
                    os.rename(old_path, new_path); wx.MessageBox("Berhasil mengganti nama file.", "Sukses",
                                                                 wx.OK | wx.ICON_INFORMATION)
                    self.UpdateFileList(self.current_dir)
                except Exception as e:
                    wx.MessageBox(f"Gagal mengganti nama:\n{e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

class PatientForm(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent); vbox = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(4, 2, 8, 8); grid.AddGrowableCol(1, 1)
        code_row = wx.BoxSizer(wx.HORIZONTAL); self.txt_code = wx.TextCtrl(self); self.btn_load = wx.Button(self, label="Load dari ID")
        code_row.Add(self.txt_code, 1, wx.EXPAND | wx.RIGHT, 6); code_row.Add(self.btn_load, 0)
        self.txt_name = wx.TextCtrl(self); self.txt_birth = wx.TextCtrl(self, value="YYYY-MM-DD")
        self.choice_gender = wx.Choice(self, choices=["", "Laki-laki", "Perempuan"])
        grid.Add(wx.StaticText(self, label="ID Pasien / Kode:"), 0, wx.ALIGN_CENTER_VERTICAL); grid.Add(code_row, 1, wx.EXPAND)
        grid.Add(wx.StaticText(self, label="Nama:"), 0, wx.ALIGN_CENTER_VERTICAL); grid.Add(self.txt_name, 1, wx.EXPAND)
        grid.Add(wx.StaticText(self, label="Tanggal Lahir:"), 0, wx.ALIGN_CENTER_VERTICAL); grid.Add(self.txt_birth, 1, wx.EXPAND)
        grid.Add(wx.StaticText(self, label="Jenis Kelamin:"), 0, wx.ALIGN_CENTER_VERTICAL); grid.Add(self.choice_gender, 0, wx.EXPAND)
        vbox.Add(grid, 0, wx.EXPAND | wx.ALL, 10)
        info = wx.StaticText(self, label="Isi identitas pasien sebelum merekam."); vbox.Add(info, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)
        self.btn_load.Bind(wx.EVT_BUTTON, self.OnLoadByCode); self.SetSizer(vbox)

    def get_patient(self):
        label = self.choice_gender.GetStringSelection().strip(); code = gender_label_to_code(label)
        return (self.txt_code.GetValue().strip(), self.txt_name.GetValue().strip(),
                self.txt_birth.GetValue().strip(), code)

    def OnLoadByCode(self, event):
        code = self.txt_code.GetValue().strip()
        if not code: wx.MessageBox("Masukkan ID Pasien/Kode terlebih dahulu.", "Info", wx.ICON_INFORMATION); return
        row = get_patient_by_code(code)
        if row is None: wx.MessageBox("Data pasien tidak ditemukan untuk ID tersebut.", "Info", wx.ICON_INFORMATION); return
        _, name, birth, gender_code = row
        if name: self.txt_name.SetValue(name)
        if birth: self.txt_birth.SetValue(birth)
        label = gender_code_to_label(gender_code)
        try: self.choice_gender.SetStringSelection(label)
        except Exception: self.choice_gender.SetSelection(0)

class AnalysisPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent); vbox = wx.BoxSizer(wx.VERTICAL)
        ctrl_row = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_row.Add(wx.StaticText(self, label="Detik:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.txt_sec = wx.TextCtrl(self, value="0.0", size=(70, -1)); ctrl_row.Add(self.txt_sec, 0, wx.ALL, 2)
        self.btn_play = wx.Button(self, label="Play"); self.btn_pause = wx.Button(self, label="Pause"); self.btn_pause.Disable()
        ctrl_row.Add((12,0)); ctrl_row.Add(self.btn_play, 0, wx.ALL, 4); ctrl_row.Add(self.btn_pause, 0, wx.ALL, 4)
        ctrl_row.Add((18,0)); self.btn_export = wx.Button(self, label="Simpan Grafik..."); ctrl_row.Add(self.btn_export, 0, wx.ALL, 4)
        self.btn_toggle = wx.Button(self, label="Tampilkan: Frekuensi (FFT)"); ctrl_row.Add(self.btn_toggle, 0, wx.ALL, 4)
        self.chk_smooth = wx.CheckBox(self, label="Smooth FFT"); ctrl_row.Add(self.chk_smooth, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        ctrl_row.AddStretchSpacer(); vbox.Add(ctrl_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.slider = wx.Slider(self, value=0, minValue=0, maxValue=1000, style=wx.SL_HORIZONTAL); vbox.Add(self.slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        self.figure, self.ax = plt.subplots(); self.canvas = FigureCanvas(self, -1, self.figure)
        self.ax.set_title('Waveform Rekaman',fontsize=20,fontweight='bold')
        self.ax.set_xlabel('Waktu (s)', fontsize=16); self.ax.set_ylabel('Amplitudo (counts)', fontsize=16)
        grid = wx.FlexGridSizer(2, 6, 6, 18)
        self.lbl_path = wx.StaticText(self, label="-"); self.lbl_peak = wx.StaticText(self, label="Peak: -")
        self.lbl_freq = wx.StaticText(self, label="Freq Dom: - Hz"); self.lbl_bpm = wx.StaticText(self, label="BPM: -")
        self.lbl_dur = wx.StaticText(self, label="Durasi: - s")
        grid.AddMany([
            (wx.StaticText(self, label="File:"), 0, wx.ALIGN_CENTER_VERTICAL),
            (self.lbl_path, 1, wx.EXPAND),
            (self.lbl_peak, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 24),
            (self.lbl_freq, 0, wx.ALIGN_CENTER_VERTICAL),
            (self.lbl_bpm, 0, wx.ALIGN_CENTER_VERTICAL),
            (self.lbl_dur, 0, wx.ALIGN_CENTER_VERTICAL),
        ])
        grid.AddGrowableCol(1, 1)
        vbox.Add(grid, 0, wx.EXPAND | wx.ALL, 8); vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 8)
        pgrid = wx.FlexGridSizer(2, 6, 6, 16); pgrid.AddGrowableCol(1,1); pgrid.AddGrowableCol(3,1); pgrid.AddGrowableCol(5,1)
        self.lbl_pcode = wx.StaticText(self, label="-"); self.lbl_pname = wx.StaticText(self, label="-")
        self.lbl_pbirth = wx.StaticText(self, label="-"); self.lbl_pgender = wx.StaticText(self, label="-"); self.lbl_pdate = wx.StaticText(self, label="-")
        pgrid.Add(wx.StaticText(self, label="Kode Pasien:"),0,wx.ALIGN_CENTER_VERTICAL); pgrid.Add(self.lbl_pcode,1,wx.EXPAND)
        pgrid.Add(wx.StaticText(self, label="Nama:"),0,wx.ALIGN_CENTER_VERTICAL); pgrid.Add(self.lbl_pname,1,wx.EXPAND)
        pgrid.Add(wx.StaticText(self, label="Jenis Kelamin:"),0,wx.ALIGN_CENTER_VERTICAL); pgrid.Add(self.lbl_pgender,1,wx.EXPAND)
        pgrid.Add(wx.StaticText(self, label="Tgl Lahir:"),0,wx.ALIGN_CENTER_VERTICAL); pgrid.Add(self.lbl_pbirth,1,wx.EXPAND)
        pgrid.Add(wx.StaticText(self, label="Tgl Periksa:"),0,wx.ALIGN_CENTER_VERTICAL); pgrid.Add(self.lbl_pdate,1,wx.EXPAND)
        pgrid.Add((10,10)); pgrid.Add((10,10)); vbox.Add(pgrid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self._plot_mode = 'time'  # 'time' atau 'freq'
        self._audio = None; self._rate = None; self._time_axis = None
        self.SetSizer(vbox)

        self._pa = pyaudio.PyAudio(); self._play_thread = None; self._play_stop = False; self._paused = False
        self._wav_path = None; self._wave_params = None; self._play_pos_frames = 0; self._duration = 0.0
        self.chk_smooth.Bind(wx.EVT_CHECKBOX, lambda e: self.update_plot())
        self.btn_toggle.Bind(wx.EVT_BUTTON, self.on_toggle_view)
        self._seek_pending = False; self._updating_slider = False
        self.btn_play.Bind(wx.EVT_BUTTON, self.on_play); self.btn_pause.Bind(wx.EVT_BUTTON, self.on_pause)
        self.slider.Bind(wx.EVT_SLIDER, self.on_seek_slider); self.btn_export.Bind(wx.EVT_BUTTON, self.on_export_graph)

    def on_export_graph(self, event):
        if not self._wav_path or not os.path.isfile(self._wav_path):
            wx.MessageBox("Tidak ada file WAV yang dimuat.", "Info", wx.ICON_INFORMATION); return
        base_dir = os.path.dirname(self._wav_path) if os.path.isdir(os.path.dirname(self._wav_path)) else os.getcwd()
        base_name = os.path.splitext(os.path.basename(self._wav_path))[0] + "_plot.png"
        wildcard = "PNG (*.png)|*.png|PDF (*.pdf)|*.pdf|SVG (*.svg)|*.svg|JPEG (*.jpg;*.jpeg)|*.jpg;*.jpeg"
        with wx.FileDialog(self, "Simpan Grafik", defaultDir=base_dir, defaultFile=base_name,
                           wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK: return
            out_path = dlg.GetPath()
        try:
            self.figure.tight_layout(); ext = os.path.splitext(out_path)[1].lower()
            if ext in [".jpg",".jpeg",".png"]: self.figure.savefig(out_path, dpi=200)
            else: self.figure.savefig(out_path)
            wx.MessageBox(f"Grafik tersimpan:\n{out_path}", "Sukses", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Gagal menyimpan grafik:\n{e}", "Error", wx.ICON_ERROR)


    def on_toggle_view(self, event):
        self._plot_mode = 'freq' if self._plot_mode == 'time' else 'time'
        self.btn_toggle.SetLabel("Tampilkan: Waktu" if self._plot_mode == 'freq' else "Tampilkan: Frekuensi (FFT)")
        self.update_plot()

    def update_plot(self):
        if self._audio is None or self._rate is None or self._time_axis is None:
            return
        if self._plot_mode == 'time':
            self._plot_time()
        else:
            self._plot_freq()

    def _plot_time(self):
        audio = self._audio; t = self._time_axis; duration = (t[-1] if t.size>0 else 0.0)
        self.ax.clear(); self.ax.plot(t, audio)
        self.ax.set_title('Waveform Rekaman',fontsize=20,fontweight='bold')
        self.ax.set_xlabel('Waktu (s)', fontsize=16); self.ax.set_ylabel('Amplitudo (counts)', fontsize=16)
        self.ax.set_xlim(0, max(duration, 1e-3)); self.ax.set_ylim(-7500, 7501)
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout(); self.canvas.draw_idle()

    def _plot_freq(self):
        x = self._audio.astype(float)
        if x.size == 0 or not self._rate or self._rate <= 0: return
        win = np.hanning(x.size); y = x*win
        Y = np.fft.rfft(y); f = np.fft.rfftfreq(y.size, d=1.0/self._rate); mag = np.abs(Y)
        eps = 1e-12; mag_db = 20.0*np.log10(mag + eps)

        if hasattr(self, 'chk_smooth') and self.chk_smooth.IsChecked():
            window_len = 125
            if mag_db.size > window_len:
                w = np.ones(window_len) / window_len
                mag_db = np.convolve(mag_db, w, mode='same')
    
        band = (f>=0) & (f<=1000.0); f_plot = f[band]; m_plot = mag_db[band]
        self.ax.clear(); self.ax.plot(f_plot, m_plot, color="#B82E1E", linewidth=1.5)
        self.ax.set_xscale('log'); self.ax.set_xlim(20,1000)
        target_ticks = [20,30,40,50,60,80,100,200,300,400,500,600,1000]
        self.ax.set_xticks(target_ticks); self.ax.get_xaxis().set_major_formatter(ScalarFormatter())
        self.ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        
        self.ax.tick_params(axis='both', which='major', labelsize=11)
        self.ax.grid(True, which='both',linestyle='--', alpha=0.5)
        self.ax.set_title('Spektrum Frekuensi (Skala Logaritmik)', fontsize=20,fontweight='bold') 
        self.ax.set_xlabel('Frekuensi (Hz)', fontsize=16); self.ax.set_ylabel('Magnitudo (dB)',fontsize=16)

        try:
            txt = self.lbl_freq.GetLabel()
            fdom = float(txt.split(':')[1].strip().split()[0])
            if 0 <= fdom <= 1000:
                self.ax.axvline(fdom, linestyle='-', linewidth=1.5, color='blue', alpha=0.7)
                y_min, y_max = self.ax.get_ylim()
                text_y_pos = y_min + 0.99 * (y_max - y_min) 

                self.ax.text(fdom, text_y_pos, f" {fdom:.1f} Hz", color='blue',
                             fontweight='bold', ha='left', va='top', fontsize=10)
        except Exception:
            pass
        self.figure.tight_layout(); self.canvas.draw_idle()

    def on_seek_slider(self, event):
        if not self._wave_params or self._duration <= 0 or self._updating_slider: return
        pos_sec = (self.slider.GetValue()/1000.0)*self._duration; fr = self._wave_params[2]
        self._play_pos_frames = int(max(0, min(pos_sec*fr, self._duration*fr)))
        if self._play_thread and self._play_thread.is_alive() and not self._paused: self._seek_pending = True
        self.txt_sec.SetValue(f"{pos_sec:.2f}")

    def on_play(self, event):
        if not self._wav_path or not os.path.isfile(self._wav_path):
            wx.MessageBox("Tidak ada file WAV yang dimuat.", "Info", wx.ICON_INFORMATION); return
        if self._paused:
            self._paused = False; self.btn_play.Disable(); self.btn_pause.Enable(); return
        try: sec = float(self.txt_sec.GetValue())
        except ValueError: sec = 0.0
        sec = max(0.0, min(sec, self._duration if self._duration>0 else sec))
        if self._wave_params: self._play_pos_frames = int(sec * self._wave_params[2])
        if self._play_thread and self._play_thread.is_alive(): return
        self._play_stop = False; self._paused = False; self.btn_play.Disable(); self.btn_pause.Enable(); self._start_worker()

    def _start_worker(self):
        def _worker():
            stream = None; wf = None
            try:
                wf = wave.open(self._wav_path, 'rb'); total_frames = wf.getnframes(); fr = wf.getframerate()
                start = max(0, min(self._play_pos_frames, total_frames)); wf.setpos(start)
                stream = self._pa.open(format=self._pa.get_format_from_width(wf.getsampwidth()),
                                       channels=wf.getnchannels(), rate=fr, output=True,
                                       frames_per_buffer=FRAMES_PER_BUFFER)
                while not self._play_stop:
                    if self._paused: time.sleep(0.05); continue
                    if self._seek_pending:
                        try: wf.setpos(max(0, min(self._play_pos_frames, total_frames)))
                        except Exception: pass
                        self._seek_pending = False
                    data = wf.readframes(FRAMES_PER_BUFFER)
                    if not data: break
                    stream.write(data)
                    cur_pos = wf.tell() / max(1, fr)
                    if self._duration > 0:
                        sval = int((cur_pos/self._duration)*1000.0); sval = max(0, min(sval, 1000))
                        def _set_slider(v=sval):
                            self._updating_slider = True
                            try: self.slider.SetValue(v)
                            finally: self._updating_slider = False
                        wx.CallAfter(_set_slider)
                    self._play_pos_frames = wf.tell()
            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Gagal memutar audio:\n{e}", "Error", wx.ICON_ERROR)
            finally:
                try:
                    if stream is not None: stream.stop_stream(); stream.close()
                except Exception: pass
                try:
                    if wf is not None: wf.close()
                except Exception: pass
                wx.CallAfter(self._finish_playback_ui)
        self._play_thread = Thread(target=_worker, daemon=True); self._play_thread.start()

    def on_pause(self, event):
        self._paused = True; self.btn_play.Enable(); self.btn_pause.Disable()

    def _finish_playback_ui(self):
        self.btn_play.Enable(); self.btn_pause.Disable(); self._paused = False; self._play_stop = True; self._play_thread = None
        if self._wav_path and os.path.isfile(self._wav_path):
            try:
                with wave.open(self._wav_path, 'rb') as wf:
                    if self._play_pos_frames >= wf.getnframes():
                        self._play_pos_frames = 0
                        def _reset_slider():
                            self._updating_slider = True
                            try: self.slider.SetValue(0)
                            finally: self._updating_slider = False
                        wx.CallAfter(_reset_slider)
            except Exception: self._play_pos_frames = 0

    def load_wav(self, path:str):
        if not os.path.isfile(path): wx.MessageBox("File tidak ditemukan", "Error", wx.ICON_ERROR); return
        try:
            wf = wave.open(path, 'rb'); fr = wf.getframerate(); nf = wf.getnframes()
            audio = np.frombuffer(wf.readframes(nf), dtype=np.int16); self._wav_path = path
            self._wave_params = (wf.getnchannels(), wf.getsampwidth(), fr); wf.close()
        except Exception as e:
            wx.MessageBox(f"Gagal membaca WAV:\n{e}", "Error", wx.ICON_ERROR); return
        duration = nf/fr if fr else 0.0; self._duration = float(duration)
        t = np.linspace(0, max(duration, 1e-6), num=len(audio)) if len(audio) else np.array([0])
        # simpan untuk digunakan oleh plot waktu/frekuensi
        self._audio = audio; self._rate = fr; self._time_axis = t
        peak, dom = compute_peak_and_freq(audio.astype(float), fr)
        bpm, _beats = compute_bpm_and_beats(audio.astype(float), fr)
        # gambar sesuai mode aktif
        self.update_plot()
        self.lbl_path.SetLabel(path); self.lbl_peak.SetLabel(f"Peak: {peak:.0f} (counts)"); self.lbl_freq.SetLabel(f"Freq Dom: {dom:.1f} Hz")
        self.lbl_bpm.SetLabel(f"BPM: {bpm:.1f}")
        self.lbl_dur.SetLabel(f"Durasi: {duration:.2f} s")
        self._play_pos_frames = 0; self._paused = False; self._play_stop = True; self._seek_pending = False
        self._updating_slider = True
        try: self.slider.SetValue(0); self.slider.Enable(self._duration > 0)
        finally: self._updating_slider = False
        self.btn_play.Enable(); self.btn_pause.Disable()
        try:
            meta = get_recording_meta_by_wav(os.path.abspath(path)) or get_recording_meta_by_wav(path)
            if meta:
                pcode, pname, pbirth, pgender_code, pdate, bpm_db = meta
                self.lbl_pcode.SetLabel(pcode or "-"); self.lbl_pname.SetLabel(pname or "-")
                self.lbl_pbirth.SetLabel(pbirth or "-"); self.lbl_pgender.SetLabel(gender_code_to_label(pgender_code) or "-")
                self.lbl_pdate.SetLabel(pdate or "-")
                # if DB already has bpm, prefer showing that (from live calc)
                if bpm_db is not None and bpm_db > 0:
                    self.lbl_bpm.SetLabel(f"BPM: {bpm_db:.1f}")
            else:
                self.lbl_pcode.SetLabel("-"); self.lbl_pname.SetLabel("-"); self.lbl_pbirth.SetLabel("-"); self.lbl_pgender.SetLabel("-"); self.lbl_pdate.SetLabel("-")
        except Exception:
            self.lbl_pcode.SetLabel("-"); self.lbl_pname.SetLabel("-"); self.lbl_pbirth.SetLabel("-"); self.lbl_pgender.SetLabel("-"); self.lbl_pdate.SetLabel("-")

class LiveWaveFormPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent); self.pa = pyaudio.PyAudio(); self.lock = Lock(); self.running = True
        self.rec_audio = []; self.rec_time = []; self.elapse = 0.0
        self.is_recording = False; self.frames = []; self.monitor_maxlen = int(RATE / FRAMES_PER_BUFFER * 5); self.monitor_audio = deque(maxlen=self.monitor_maxlen * FRAMES_PER_BUFFER)
        self.stream_record = None; self.base_dir = None
        self.figure, self.ax = plt.subplots(); self.line, = self.ax.plot([], [], lw=1.25)
        self.ax.set_ylim([-7500, 7501]); self.ax.set_xlim(0, 5)
        self.ax.set_title('Live Audio Waveform (Recording)',fontsize=20,fontweight='bold')
        self.ax.set_xlabel('Waktu (s)', fontsize=16); self.ax.set_ylabel('Amplitudo (counts)', fontsize=16)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.start_btn = wx.Button(self, label='Start Recording'); self.stop_btn = wx.Button(self, label='Stop Recording'); self.stop_btn.Disable()
        self.chk_video = wx.CheckBox(self, label='Buat video visual (butuh ffmpeg)')
        self.lbl_result_peak = wx.StaticText(self, label="Peak: -"); self.lbl_result_freq = wx.StaticText(self, label="Freq Dom: - Hz")
        self.lbl_result_bpm = wx.StaticText(self, label="BPM: -")
        self.start_btn.Bind(wx.EVT_BUTTON, self.OnStart); self.stop_btn.Bind(wx.EVT_BUTTON, self.OnStop)
        top_row = wx.BoxSizer(wx.HORIZONTAL); top_row.AddStretchSpacer(); top_row.Add(self.start_btn,0,wx.ALL,5)
        top_row.Add(self.stop_btn,0,wx.ALL,5); top_row.Add(self.chk_video,0,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5); top_row.AddStretchSpacer()
        result_row = wx.BoxSizer(wx.HORIZONTAL); result_row.AddStretchSpacer(); result_row.Add(self.lbl_result_peak,0,wx.ALL,5)
        result_row.Add((12,0)); result_row.Add(self.lbl_result_freq,0,wx.ALL,5); result_row.Add((12,0)); result_row.Add(self.lbl_result_bpm,0,wx.ALL,5); result_row.AddStretchSpacer()
        sizer = wx.BoxSizer(wx.VERTICAL); sizer.Add(top_row,0,wx.EXPAND|wx.ALL,5); sizer.Add(self.canvas,1,wx.EXPAND|wx.ALL,8); sizer.Add(result_row,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM,8); self.SetSizer(sizer)
        self.timer = wx.Timer(self); self.Bind(wx.EVT_TIMER, self.OnUpdate, self.timer); self.timer.Start(50)
        self.stream_in = self.pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
        self.stream_out = self.pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=FRAMES_PER_BUFFER)
        self.rt_thread = Thread(target=self.StreamLoop, daemon=True); self.rt_thread.start()

    def shutdown(self):
        self.running = False
        try:
            if self.is_recording: self.is_recording = False
        except Exception: pass
        try: self.timer.Stop()
        except Exception: pass
        try:
            if self.stream_in is not None: self.stream_in.stop_stream(); self.stream_in.close()
        except Exception: pass
        try:
            if self.stream_out is not None: self.stream_out.stop_stream(); self.stream_out.close()
        except Exception: pass
        try: self.pa.terminate()
        except Exception: pass

    def StreamLoop(self):
        while self.running:
            try:
                data = self.stream_in.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                self.stream_out.write(data) # Audio monitoring (suara keluar speaker)
            except Exception as e:
                print(f"[Audio Error] {e}"); continue
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            with self.lock:
                self.monitor_audio.extend(audio_data)

                if self.is_recording:
                    self.frames.append(data)
                    self.rec_audio.extend(audio_data)
                    
                    dt = FRAMES_PER_BUFFER / RATE

                    times = np.linspace(self.elapse, self.elapse + dt, num=FRAMES_PER_BUFFER, endpoint=False)
                    self.rec_time.extend(times)
                    self.elapse += dt

    def OnUpdate(self, event):
        with self.lock:
            if self.is_recording:
                if not self.rec_audio: return
                
                y_data = list(self.rec_audio)
                x_data = list(self.rec_time)
                
                self.line.set_data(x_data, y_data)
                
                current_time = self.elapse
                if current_time > 5:
                    self.ax.set_xlim(current_time - 5, current_time)
                else:
                    self.ax.set_xlim(0, 5)

            else:
                if not self.monitor_audio: return
                
                y_data = list(self.monitor_audio)
                x_data = np.linspace(0, 5, num=len(y_data))
                
                self.line.set_data(x_data, y_data)
                self.ax.set_xlim(0, 5) 
        
        self.canvas.draw_idle()

    def ensure_output_dir(self):
        if self.base_dir and os.path.isdir(self.base_dir): return True
        with wx.DirDialog(self, "Pilih folder untuk menyimpan hasil:", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL: return False
            self.base_dir = dlg.GetPath(); return True

    def OnStart(self, event):
        parent = self.GetParent(); frame = parent.GetParent()
        pcode = pname = pbirth = pgender_code = ""
        if hasattr(frame, 'patient_form'):
            pcode, pname, pbirth, pgender_code = frame.patient_form.get_patient()
            if pcode and (not pname or not pbirth or not pgender_code):
                row = get_patient_by_code(pcode)
                if row is not None:
                    _, name_db, birth_db, gender_db = row
                    if not pname and name_db: pname = name_db; frame.patient_form.txt_name.SetValue(name_db)
                    if not pbirth and birth_db: pbirth = birth_db; frame.patient_form.txt_birth.SetValue(birth_db)
                    if not pgender_code and gender_db:
                        try: frame.patient_form.choice_gender.SetStringSelection(gender_code_to_label(gender_db))
                        except Exception: pass
        if not pcode or not pname:
            wx.MessageBox("Lengkapi ID Pasien dan Nama sebelum memulai live plotting/rekaman.", "Data Pasien", wx.ICON_WARNING); return
        if not self.ensure_output_dir(): return
        
        with self.lock:
            self.is_recording = True; self.frames = []; self.rec_audio = []   # Reset data audio rekaman
            self.rec_time = []    # Reset sumbu waktu rekaman
            self.elapse = 0.0     # Reset counter waktu ke 0
        
        # Reset Label GUI
        self.lbl_result_peak.SetLabel("Peak: -")
        self.lbl_result_freq.SetLabel("Freq Dom: - Hz")
        self.lbl_result_bpm.SetLabel("BPM: -")
        
        self.start_btn.Disable()
        self.stop_btn.Enable()

    def OnStop(self, event):
        if not self.is_recording: return
        self.is_recording = False; self.start_btn.Enable(); self.stop_btn.Disable()
        audio_bytes = b''.join(self.frames); audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        duration = float(len(audio_array)/RATE) if audio_array.size else 0.0
        peak_amp, dom_freq = compute_peak_and_freq(audio_array.astype(float), RATE)
        bpm, beat_times = compute_bpm_and_beats(audio_array, RATE)
        self.lbl_result_peak.SetLabel(f"Peak: {peak_amp:.0f}"); self.lbl_result_freq.SetLabel(f"Freq Dom: {dom_freq:.1f} Hz")
        self.lbl_result_bpm.SetLabel(f"BPM: {bpm:.1f}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_filename = f'sample_{timestamp}.wav'; png_filename = f'sample_{timestamp}.png'; vis_filename = f'sample_{timestamp}_visual.mp4'; out_filename = f'sample_{timestamp}.mp4'
        audio_dir = os.path.join(self.base_dir, 'Hasil Audio'); graph_dir = os.path.join(self.base_dir, 'Hasil Grafik')
        video_dir = os.path.join(self.base_dir, 'Hasil Video'); visual_dir = os.path.join(self.base_dir, 'Hasil Visual')
        for d in (audio_dir, graph_dir, video_dir, visual_dir): os.makedirs(d, exist_ok=True)
        wav_path = os.path.join(audio_dir, wav_filename); png_path = os.path.join(graph_dir, png_filename)
        video_path = os.path.join(visual_dir, vis_filename); final_path = os.path.join(video_dir, out_filename)
        wf = wave.open(wav_path, 'wb'); wf.setnchannels(CHANNELS); wf.setsampwidth(self.pa.get_sample_size(FORMAT)); wf.setframerate(RATE); wf.writeframes(audio_bytes); wf.close()
        times = np.linspace(0, duration, num=len(audio_array)) if audio_array.size>0 else np.array([0])
        fig, ax = plt.subplots(figsize=(12,4)); ax.plot(times, audio_array); ax.set_title('Waveform Audio', fontsize=20,fontweight='bold')
        ax.set_xlabel('Waktu (s)', fontsize=16); ax.set_ylabel('Amplitudo (counts)', fontsize=16); ax.set_xlim(0, max(duration, 1e-3)); ax.set_ylim(-7500,7501); ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(png_path); plt.close(fig)
        made_video = False
        if self.chk_video.IsChecked():
            try: create_waveform_video(wav_path, video_path); merge_audio_video(video_path, wav_path, final_path); made_video = True
            except Exception as e: wx.MessageBox(f"Gagal membuat video visual:\n{e}", "FFmpeg Error", wx.ICON_WARNING)
        parent = self.GetParent(); frame = parent.GetParent()
        if hasattr(frame, 'patient_form'):
            pcode, pname, pbirth, pgender_code = frame.patient_form.get_patient()
            if not pcode or not pname:
                wx.MessageBox("Lengkapi ID Pasien dan Nama sebelum merekam!", "Data Pasien", wx.ICON_WARNING)
            else:
                pid = upsert_patient(pcode, pname, pbirth, pgender_code)
                insert_recording(pid, time.strftime("%Y-%m-%d %H:%M:%S"), wav_path, png_path, (final_path if made_video else ''),
                                 duration, peak_amp, dom_freq, bpm)
                frame.refresh_recent()
        msg = f"Audio & grafik disimpan.\n\nDurasi: {duration:.2f} s\nPeak amplitude: {peak_amp:.0f}\nDominant freq: {dom_freq:.1f} Hz\nBPM (estimasi): {bpm:.1f}"
        wx.MessageBox(msg, "Selesai", wx.OK | wx.ICON_INFORMATION)

def create_waveform_video(wav_path: str, video_path: str, fps: int = 30):
    wf = wave.open(wav_path, 'rb'); framerate = wf.getframerate(); nframes = wf.getnframes(); audio_bytes = wf.readframes(nframes); wf.close()
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16); duration = nframes/framerate if framerate else 0
    total_frames = max(1, int(duration*fps)); samples_per_frame = max(1, int(len(audio_array)/total_frames))
    times = np.linspace(0, duration, num=len(audio_array)) if len(audio_array) else np.array([0])
    fig, ax = plt.subplots(figsize=(32,4)); ax.set_xlim(0, max(duration, 1e-3)); ax.set_ylim(-7500,7501); line, = ax.plot([], [], lw=2)
    def init(): line.set_data([], []); return line,
    def animate(i): end = (i+1)*samples_per_frame; x = times[:end]; y = audio_array[:end]; line.set_data(x,y); return line,
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=1000/fps, blit=True)
    ani.save(video_path, writer='ffmpeg', fps=fps); plt.close(fig)

def merge_audio_video(video_path: str, wav_path: str, final_output: str):
    cmd = ['ffmpeg','-y','-i',video_path,'-i',wav_path,'-c:v','copy','-c:a','aac','-shortest',final_output]
    subprocess.run(cmd, check=True)

class RecentPanel(wx.Panel):
    def __init__(self, parent, on_open_wav=None):
        super().__init__(parent); self.on_open_wav = on_open_wav; self.rowdata = {}
        vbox = wx.BoxSizer(wx.VERTICAL); style = wx.LC_REPORT | wx.BORDER_SUNKEN | wx.WANTS_CHARS
        self.list = wx.ListCtrl(self, style=style)
        for i, (hdr, w) in enumerate([("ID",60),("Kode Pasien",140),("Nama",160),("JK",70),("Tanggal Periksa",170),
                                      ("Durasi (s)",90),("Peak",80),("Freq Dom (Hz)",110),("BPM",80),("WAV",360)]):
            self.list.InsertColumn(i, hdr, width=w)
        vbox.Add(self.list, 1, wx.EXPAND | wx.ALL, 8)
        btnrow = wx.BoxSizer(wx.HORIZONTAL); self.btn_refresh = wx.Button(self, label="Refresh")
        self.btn_delete = wx.Button(self, label="Delete"); self.btn_delete.Disable()
        btnrow.Add(self.btn_refresh,0,wx.ALL,5); btnrow.Add(self.btn_delete,0,wx.ALL,5); vbox.Add(btnrow,0,wx.ALIGN_LEFT)
        self.SetSizer(vbox)
        self.btn_refresh.Bind(wx.EVT_BUTTON, lambda evt: self.populate())
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.open_wav)
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_select)
        self.list.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_deselect)
        self.list.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.btn_delete.Bind(wx.EVT_BUTTON, self.on_delete_clicked)

    def on_select(self, event): self.btn_delete.Enable()
    def on_deselect(self, event):
        if self.list.GetFirstSelected() == -1: self.btn_delete.Disable()
    def on_key_down(self, event):
        if event.GetKeyCode() == wx.WXK_DELETE: self.delete_selected()
        else: event.Skip()
    def on_delete_clicked(self, event): self.delete_selected()

    def _iter_selected_indices(self):
        idx = self.list.GetFirstSelected()
        while idx != -1:
            yield idx
            idx = self.list.GetNextItem(idx, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)

    def delete_selected(self):
        selected = list(self._iter_selected_indices())
        if not selected: return
        lines = []
        for idx in selected:
            rec_id = self.list.GetItemText(idx, 0); wavp = self.list.GetItemText(idx, 9)
            lines.append(f"- ID {rec_id}: {os.path.basename(wavp) if wavp else '(tanpa WAV)'}")
        msg = "Hapus rekaman terpilih dari database?\n\n" + "\n".join(lines) + \
              "\n\nFile terkait (WAV/PNG/VIDEO) juga akan dihapus dari disk (jika ada)."
        if wx.MessageBox(msg, "Konfirmasi Hapus", wx.YES_NO | wx.ICON_WARNING) != wx.YES: return
        errors = []
        for idx in selected:
            rec_id = self.list.GetItemText(idx, 0); wavp = self.list.GetItemText(idx, 9)
            data = self.rowdata.get(idx, {}); pngp = data.get('png_path',''); vidp = data.get('video_path','')
            if not pngp or not vidp or not wavp:
                dbpaths = get_recording_paths_by_id(int(rec_id))
                if dbpaths:
                    if not wavp: wavp = dbpaths[0]
                    if not pngp: pngp = dbpaths[1]
                    if not vidp: vidp = dbpaths[2]
            for path in [wavp, pngp, vidp]:
                if path and os.path.isfile(path):
                    try: os.remove(path)
                    except Exception as e: errors.append(f"Gagal menghapus file: {path}\n{e}")
            try: delete_recording_by_id(int(rec_id))
            except Exception as e: errors.append(f"Gagal hapus DB untuk ID {rec_id}:\n{e}")
        self.populate()
        if errors:
            wx.MessageBox("Selesai, tetapi ada peringatan:\n\n" + "\n\n".join(errors), "Selesai (dengan peringatan)", wx.ICON_WARNING)
        else:
            wx.MessageBox("Rekaman berhasil dihapus (DB + file).", "Sukses", wx.OK | wx.ICON_INFORMATION)

    def populate(self):
        self.rowdata.clear(); self.list.DeleteAllItems()
        rows = fetch_recent_recordings(limit=200)
        for r in rows:
            rid, pcode, pname, pgender_code, dt, wavp, pngp, vidp, dur, peak, df, bpm = r
            idx = self.list.InsertItem(self.list.GetItemCount(), str(rid))
            self.list.SetItem(idx, 1, pcode or ''); self.list.SetItem(idx, 2, pname or '')
            self.list.SetItem(idx, 3, (pgender_code or '')); self.list.SetItem(idx, 4, dt or '')
            self.list.SetItem(idx, 5, f"{dur:.2f}" if dur is not None else ''); self.list.SetItem(idx, 6, f"{peak:.0f}" if peak is not None else '')
            self.list.SetItem(idx, 7, f"{df:.1f}" if df is not None else ''); self.list.SetItem(idx, 8, f"{bpm:.1f}" if bpm is not None else '')
            self.list.SetItem(idx, 9, wavp or '')
            self.rowdata[idx] = {'rec_id': rid, 'wav_path': wavp or '', 'png_path': pngp or '', 'video_path': vidp or ''}
        self.btn_delete.Enable(self.list.GetFirstSelected() != -1)

    def open_wav(self, event):
        idx = event.GetIndex(); wavp = self.list.GetItemText(idx, 9)
        if not wavp or not os.path.isfile(wavp):
            wx.MessageBox("File WAV tidak ditemukan.", "Info", wx.ICON_INFORMATION); return
        if callable(self.on_open_wav): self.on_open_wav(wavp)
        else:
            try:
                if sys.platform.startswith('win'): os.startfile(wavp)
                elif sys.platform == 'darwin': subprocess.run(['open', wavp])
                else: subprocess.run(['xdg-open', wavp])
            except Exception as e:
                wx.MessageBox(f"Gagal membuka file:\n{e}", "Error", wx.ICON_ERROR)

class AudioRecorder(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1240, 880))
        init_db()
        menubar = wx.MenuBar(); filemenu = wx.Menu()
        export_item = filemenu.Append(wx.ID_ANY, "Ekspor Data (CSV)...\tCtrl+E", "Ekspor patients/recordings ke CSV")
        filemenu.AppendSeparator(); exit_item = filemenu.Append(wx.ID_EXIT, "Keluar\tCtrl+W", "Tutup aplikasi")
        menubar.Append(filemenu, "&File"); self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnExportCSV, export_item); self.Bind(wx.EVT_MENU, self.OnClose, exit_item)

        self.notebook = wx.Notebook(self)
        self.analysis_panel = AnalysisPanel(self.notebook)
        self.explorer_panel = ExplorerPanel(self.notebook, on_open_wav=self.open_in_app)
        self.patient_form = PatientForm(self.notebook)
        self.live_waveform = LiveWaveFormPanel(self.notebook)
        self.recent_panel = RecentPanel(self.notebook, on_open_wav=self.open_in_app)

        self.notebook.AddPage(self.explorer_panel, "Explorer")
        self.notebook.AddPage(self.patient_form, "Data Pasien")
        self.notebook.AddPage(self.live_waveform, "Live Plotting & Rekam")
        self.notebook.AddPage(self.recent_panel, "Riwayat Rekaman")
        self.notebook.AddPage(self.analysis_panel, "Analisis Rekaman")

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Centre(); self.Show(); wx.CallAfter(self.refresh_recent)

    def refresh_recent(self):
        try: self.recent_panel.populate()
        except Exception: pass

    def open_in_app(self, wav_path: str):
        try:
            self.analysis_panel.load_wav(wav_path)
            try: idx = self.notebook.GetPageIndex(self.analysis_panel)
            except AttributeError:
                idx = -1
                for i in range(self.notebook.GetPageCount()):
                    if self.notebook.GetPage(i) is self.analysis_panel: idx = i; break
            if idx != wx.NOT_FOUND and idx >= 0: self.notebook.SetSelection(idx)
        except Exception as e:
            wx.MessageBox(f"Gagal memuat rekaman:\n{e}", "Error", wx.ICON_ERROR)

    def OnExportCSV(self, event):
        with wx.DirDialog(self, "Pilih folder tujuan untuk CSV:", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK: return
            out_dir = dlg.GetPath()
        try:
            p_csv, r_csv, j_csv = export_db_to_csv(out_dir)
            wx.MessageBox(f"Ekspor selesai:\n- {p_csv}\n- {r_csv}\n- {j_csv}", "Sukses", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Gagal mengekspor CSV:\n{e}", "Error", wx.ICON_ERROR)

    def OnClose(self, event):
        try: self.live_waveform.shutdown()
        except Exception: pass
        self.Destroy()


if __name__ == '__main__':
    app = wx.App(False)
    frame = AudioRecorder(None, 'Rekaman dan Analisis Suara Auskultasi Jantung')
    app.SetExitOnFrameDelete(True)
    app.MainLoop()