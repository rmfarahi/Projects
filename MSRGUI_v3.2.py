# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 22:39:13 2022

@author: FARAHROX000
"""

import tkinter as tk
from tkinter.filedialog import askopenfile
import os
import pandas as pd
import numpy as np
import CSV_IO
import scipy.signal as ss
from copy import copy, deepcopy
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

global x_t, x_t_cut, x_t_cut_new
global y_p, y_p_cut, y_p_cut_new
global y_p_fit, yoff
global cut_len

class data_matrix:
    
    def __init__(self, x,y):
        data_matrix.x = x
        data_matrix.y = y
        data_matrix.abc = [0,0,0]
        data_matrix.n_data = np.shape(data_matrix.x)[0]
        print("num data points", data_matrix.n_data)
        data_matrix.i0 = 0  #auto index found
        data_matrix.i1 = 0
        
    def cut(cut1, cut2):
        data_matrix.cut1 = int(cut1)
        data_matrix.cut2 = int(cut2)
        data_matrix.cut_len = cut2-cut1
        x_cut = np.zeros(data_matrix.cut_len)
        y_cut = np.zeros(data_matrix.cut_len)
        x_cut[0:data_matrix.cut_len] = data_matrix.x[cut1:cut2]
        y_cut[0:data_matrix.cut_len] = data_matrix.y[cut1:cut2]
        data_matrix.x_cut = x_cut
        data_matrix.y_cut = y_cut
        data_matrix.x_off = data_matrix.x_cut[0]
        data_matrix.y_off = min(data_matrix.y_cut)
        data_matrix.x_cut_shift = data_matrix.x_cut-data_matrix.x_off
        data_matrix.y_cut_shift = data_matrix.y_cut-data_matrix.y_off 
        
        
    def fit(f, abc):
        data_matrix.y_fit = f
        data_matrix.abc = abc
    
class m_matrix:
    n_meta = 0
    
    def __init__(self, mrow_labels, m_array):
        m_matrix.m_array = m_array
        m_matrix.labels=mrow_labels
        m_matrix.n_meta = np.shape(mrow_labels)[0]

def read_meta_file():
    m_file = askopenfile(mode ='r', filetypes=[('csv Files','*.csv')])
    if m_file is not None:
        mrow_labels, mcol_labels, m_array = CSV_IO.read_csv_v2(m_file)
        m_matrix(mrow_labels, m_array)
        # print(np.shape(m_matrix.labels), m_matrix.labels)
        # print(np.shape(m_matrix.m_array), m_matrix.m_array)
        mname.set(m_file.name)
        m_matrix(mrow_labels,m_array) ## fill metadata in the class
        update_meta()

def update_meta():
    run = np.int32(ent_run.get())
    run_ind = run-1
    gas_meta_val = m_matrix.labels[1]
    salt_meta_val = m_matrix.labels[0]
    temp_meta_val = m_matrix.m_array[7][run_ind]
    gasvol_meta_val = m_matrix.m_array[10][run_ind]
    dismole_meta_val = m_matrix.m_array[15][run_ind]
    saltvol_meta_val = m_matrix.m_array[20][run_ind]
    
    
    meta01_lbl.set("Gas")
    meta02_lbl.set("Salt")
    meta03_lbl.set(m_matrix.labels[7])
    meta04_lbl.set(m_matrix.labels[10])
    meta05_lbl.set(m_matrix.labels[15])
    meta06_lbl.set(m_matrix.labels[20])
    
    
    meta01_val.set(gas_meta_val)
    meta02_val.set(salt_meta_val)
    meta03_val.set("{0:6.0f}".format(temp_meta_val))
    meta04_val.set("{0:6.5f}".format(gasvol_meta_val))
    meta05_val.set("{0:6.3e}".format(dismole_meta_val))
    meta06_val.set("{0:6.5f}".format(saltvol_meta_val))
    
    
def update_meta_fit():
    run = np.int32(ent_run.get())
    run_ind = run-1
    t_0 =  m_matrix.m_array[28][run_ind]
    t_1 =  m_matrix.m_array[31][run_ind]
    p_0 =  m_matrix.m_array[27][run_ind]
    p_1 =  m_matrix.m_array[29][run_ind]
    deltaP =  m_matrix.m_array[30][run_ind]
    solbcm_val = m_matrix.m_array[22][run_ind]
    solbdm_val = m_matrix.m_array[23][run_ind]

    
    det01_lbl.set(m_matrix.labels[28]) #t_0
    det02_lbl.set(m_matrix.labels[31]) #t_1
    det03_lbl.set(m_matrix.labels[27]) #p_0
    det04_lbl.set(m_matrix.labels[29]) #p_1
    det05_lbl.set(m_matrix.labels[30]) #deltaP
    
    det01_val.set("{0:6.2f}".format(t_0))
    det02_val.set("{0:6.2f}".format(t_1))
    det03_val.set("{0:6.0f}".format(p_0))
    det04_val.set("{0:6.0f}".format(p_1))
    det05_val.set("{0:6.0f}".format(deltaP))
    
    
    solb01_lbl.set(m_matrix.labels[22])
    solb02_lbl.set(m_matrix.labels[23])
    solb01_val.set("{0:6.3e}".format(solbcm_val))
    solb02_val.set("{0:6.3e}".format(solbdm_val))

def read_input_file():
    global x_t, x_t_cut
    global y_p, y_p_cut
    global y_p_fit
    global cut_len

    def x_time (x):
        return x_t(x)
    
    def x_index (x):
        return x
        
        
    file = askopenfile(mode ='r', filetypes=[('csv Files','*.csv')])
    if file is not None:
        if os.path.basename(file.name).endswith('.csv'):
            data = pd.read_csv(file)
            x_t = data["Time [s]"]
            y_p = data["Diff Pressure (mbar)"]
            # print(np.shape(x_t))
            data_matrix(x_t, y_p)  ##### put data into class
            fname.set(file.name)

            p0.cla()
            p0.plot(x_t, y_p, color = 'royalblue')
            p0.set_title('Raw Data')
            p0.set_xlabel('Time (s)')
            p0.set_ylabel('Diff. Pressure (mbar)')
            # secax = p0.secondary_xaxis('top')
            # secax.set_xlabel('Index (#)')
            p0.legend(facecolor= "lightblue", loc=0)
            p0.set
            bigraw_plot.draw()    
            
            update_data()

def update_data():
    ndata = data_matrix.n_data
    tmax = data_matrix.x[ndata-1]
    data01_lbl.set("Number of data points")
    data01_val.set("{0:7.0f}".format(ndata))
    data02_lbl.set("Experiment duration (s)")
    data02_val.set("{0:6.1f}".format(tmax))
    
def cut_data():
    global start_index
    global x_t, x_t_cut, x_t_cut_new
    global y_p, y_p_cut, y_p_cut_new
    global y_p_fit, yoff  
    global cut_len
    
    update_meta()
    run = np.int32(ent_run.get())
    run_ind = run-1
    
    def x_time (x):
        return x_t(x)
    
    def x_index (x):
        return x
       
    # if ent_cut_min.get() != "":        
    #     start_index = int(ent_cut_min.get())
    # else:
    #     start_index = 0
    # if ent_cut_max.get() != "":        
    #     end_index = int(ent_cut_max.get())
    # else:
    #     end_index = 100
    start_index = data_matrix.i0
    end_index = data_matrix.i1
  
    t0 = t0_val.get()
    t1 = t1_val.get()
    
    if (t0 >= 0.0 and t1 > t0):
        start_index = np.abs(data_matrix.x-t0).argmin()
        end_index = np.abs(data_matrix.x-t1).argmin()
    else:
        start_index = 150000
        end_index = 200000
    
    print("Found cut indices:", start_index,end_index)    
    data_matrix.cut(start_index, end_index) ##### cut data in class
    
    cut_len = end_index-start_index
    x_t_cut = np.zeros(cut_len)
    y_p_cut = np.zeros(cut_len)
    x_t_cut[0:cut_len] = x_t[start_index:end_index]
    y_p_cut[0:cut_len] = y_p[start_index:end_index]
    
    # update meta matrix
    m_matrix.m_array[28][run_ind] = data_matrix.x[start_index]
    m_matrix.m_array[31][run_ind] = data_matrix.x[end_index]
    m_matrix.m_array[27][run_ind] = data_matrix.y[start_index]
    m_matrix.m_array[29][run_ind] = data_matrix.y[end_index]
    m_matrix.m_array[30][run_ind] = data_matrix.y[end_index]-data_matrix.y[start_index]
    
    yoff = min(y_p_cut)
    x_t_cut_new = x_t_cut-x_t_cut[0]
    y_p_cut_new = y_p_cut-yoff  #y_p_cut[cut_len-1]
    y_p_fit = np.zeros(cut_len)
    
    
    p0.cla()
    p0.plot(data_matrix.x, data_matrix.y, label='raw', color = 'royalblue')
    p0.plot(data_matrix.x_cut, data_matrix.y_cut, label='cut', color = 'red')
    p0.set_title('Raw Data')
    p0.set_xlabel('Time (s)')
    p0.set_ylabel('Diff. Pressure (mbar)')
    # secax = p0.secondary_xaxis('top').set_ticks(data_matrix.x)
    # secax.set_xlabel('Index (#)')
    p0.legend(facecolor="lightblue", loc=0)
    bigraw_plot.draw()
    
    start_x = data_matrix.x_cut[0]
    start_y = data_matrix.y_cut[0]
    end_x = data_matrix.x_cut[-1]
    end_y = data_matrix.y_cut[-1]
    
    p1.cla()
    p1.plot(data_matrix.x_cut, data_matrix.y_cut, label='cut', color = 'red')
    p1.plot(start_x, start_y, marker="o", markersize=6, markerfacecolor="green", markeredgecolor="green")
    p1.text(start_x+250, start_y, '(t0, P0)', ha='left', va='top')
    p1.plot(end_x, end_y, marker="o", markersize=6, markerfacecolor="green", markeredgecolor="green")
    p1.text(end_x, end_y+23, '(t*, P*)', ha='right', va='bottom')
    p1.set_title('Cut Data')
    p1.set_xlabel('Time (s)')
    p1.set_ylabel('Diff. Pressure (mbar)')
    p1.legend(facecolor= "lightblue", loc=0)
    raw_plot.draw() 

    p2.cla()
    
def board_num():
    i=0
    print(i)

def detect_device():
    i=1
    print(i)
      
def fit_analysis():
    global start_index, end_index
    global x_t, x_t_cut
    global y_p, y_p_cut
    global y_p_fit, yoff
    global x_t_cut_new, y_p_cut_new
    global cut_len
       
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c 
    
    run_num = np.int32(ent_run.get())
    
    x_t_cut_new = data_matrix.x_cut_shift  ##### get from class
    y_p_cut_new = deepcopy(data_matrix.y_cut_shift)  ##### get from class
    
    print("spike before", y_p_cut_new[208240-56880:208260-56880])
    if run_num==3:
        y_p_cut_new = ss.medfilt(y_p_cut_new, kernel_size=11)
        print("spike after", data_matrix.y[208240-56880:208260-56880])
    
    abc, cov = curve_fit(exp_func, x_t_cut_new, y_p_cut_new)
    fit_params_a.set("{0:6.5f}".format(abc[0]))
    fit_params_b.set("{0:6.5f}".format(abc[1]))
    fit_params_c.set("{0:6.5f}".format(abc[2]))
    
    update_meta_fit() ## update metadata on GUI
    
    y_p_fit = np.zeros(cut_len)
    y_p_fit = exp_func(x_t_cut_new, *abc)
    y_p_fit_new = y_p_fit + data_matrix.y_off ##### from class
    data_matrix.fit(y_p_fit_new,abc)   ##### update class
    
    print("a =", round(abc[0],4), "\tb =", round(abc[1],4), "\tc =", round(abc[2],4))
    reg_eq = round(abc[0],4),"+ e^(-",round(abc[1],4),"*x) +",round(abc[2],4)
    print(reg_eq)
    r2 = r2_score(y_p_cut, y_p_fit_new)
    val_rsq.set("{0:6.5f}".format(r2))
    r2_str = "r^2=" + str(round(r2, 4)) + " "
    print("r^2 =", r2)
    

      
    p2.cla()
    p2.plot(data_matrix.x_cut, data_matrix.y_cut, label='cut', color = 'red')
    p2.plot(data_matrix.x_cut, data_matrix.y_fit, label='fit', color = 'green')
    p2.set_title('Fit Data')
    p2.set_xlabel('Time (s)')
    p2.set_ylabel('Diff. Pressure (mbar)')
    p2.legend(facecolor= 'lightblue', title=r2_str,loc=0)
    fit_plot.draw()   

          
def get_current_slider_value():
    return int('{: .0f}'.format(val_slider.get()))

def slider_changed(event):
    n_data = data_matrix.n_data
    slider_val = get_current_slider_value()
    run_pt = np.int32((slider_val/100)*n_data)
    run_num = np.int32(ent_run.get())
    print("n_data is", n_data)
    print("slider val:", slider_val)
    print("run num:", run_num)
    print("run point: ", run_pt)
    r1_i0 = 144850
    r1_i1 = 349370
    r2_i0 = 439460
    r2_i1 = 591748
    r3_i0 = 56876
    r3_i1 = 389604
    r4_i0 = 430000
    r4_i1 = 589782
    r5_i0 = 94400
    r5_i1 = 573040
    
    if (run_num==1) and (r1_i0 <= run_pt <= r1_i1):
        print("run 1- found a match")
        data_matrix.i0 = r1_i0
        data_matrix.i1 = r1_i1
        r1_t0 = data_matrix.x[r1_i0]
        r1_t1 = data_matrix.x[r1_i1]
        t0_val.set("{0:6.0f}".format(r1_t0))
        t1_val.set("{0:6.0f}".format(r1_t1))
        
    elif (run_num==2) and (r2_i0 <= run_pt <= r2_i1):
        print("run 2- found a match")
        data_matrix.i0 = r2_i0
        data_matrix.i1 = r2_i1
        r2_t0 = data_matrix.x[r2_i0]
        r2_t1 = data_matrix.x[r2_i1]
        t0_val.set("{0:6.0f}".format(r2_t0))
        t1_val.set("{0:6.0f}".format(r2_t1))
        
    elif (run_num==3) and (r3_i0 <= run_pt <= r3_i1):
        print("run 2- found a match")
        data_matrix.i0 = r3_i0
        data_matrix.i1 = r3_i1
        r3_t0 = data_matrix.x[r3_i0]
        r3_t1 = data_matrix.x[r3_i1]
        t0_val.set("{0:6.0f}".format(r3_t0))
        t1_val.set("{0:6.0f}".format(r3_t1))
        
    elif (run_num==4) and (r4_i0 <= run_pt <= r4_i1):
        print("run 4- found a match")
        data_matrix.i0 = r4_i0
        data_matrix.i1 = r4_i1
        r4_t0 = data_matrix.x[r4_i0]
        r4_t1 = data_matrix.x[r4_i1]
        t0_val.set("{0:6.0f}".format(r4_t0))
        t1_val.set("{0:6.0f}".format(r4_t1))
    
    elif (run_num==5) and (r5_i0 <= run_pt <= r5_i1):
        print("run 5- found a match")
        data_matrix.i0 = r5_i0
        data_matrix.i1 = r5_i1
        r5_t0 = data_matrix.x[r5_i0]
        r5_t1 = data_matrix.x[r5_i1]
        t0_val.set("{0:6.0f}".format(r5_t0))
        t1_val.set("{0:6.0f}".format(r5_t1))
        
    else:
        data_matrix.i0 = 0
        data_matrix.i1 = 0
        t0_val.set(0)
        t1_val.set(0)
        print("no run match found")
        

main_window = tk.Tk()

devname=tk.StringVar()
dwdth=30
fname=tk.StringVar()
mname=tk.StringVar()
update=tk.IntVar()

val_slider = tk.DoubleVar()
val_rsq=tk.StringVar()

t0_val = tk.DoubleVar()
t1_val = tk.DoubleVar()

fit_params_a = tk.StringVar()
fit_params_b = tk.StringVar()
fit_params_c = tk.StringVar()

data01_val = tk.StringVar()
data01_lbl = tk.StringVar()
data02_val = tk.StringVar()
data02_lbl = tk.StringVar()

meta01_val = tk.StringVar()
meta01_lbl = tk.StringVar()
meta02_val = tk.StringVar()
meta02_lbl = tk.StringVar()
meta03_val = tk.DoubleVar()
meta03_lbl = tk.StringVar()
meta04_val = tk.DoubleVar()
meta04_lbl = tk.StringVar()
meta05_val = tk.DoubleVar()
meta05_lbl = tk.StringVar()
meta06_val = tk.DoubleVar()
meta06_lbl = tk.StringVar()

det01_lbl = tk.StringVar()
det01_val = tk.DoubleVar()
det02_lbl = tk.StringVar()
det02_val = tk.DoubleVar()
det03_lbl = tk.StringVar()
det03_val = tk.DoubleVar()
det04_lbl = tk.StringVar()
det04_val = tk.DoubleVar()
det05_lbl = tk.StringVar()
det05_val = tk.DoubleVar()
det06_lbl = tk.StringVar()
det06_val = tk.DoubleVar()

solb01_lbl = tk.StringVar()
solb01_val = tk.DoubleVar()
solb02_lbl = tk.StringVar()
solb02_val = tk.DoubleVar()

fwdth=100
mwdth=100
app_width = 1200
app_height = 680
screen_width = main_window.winfo_screenwidth()
screen_height = main_window.winfo_screenheight()
app_x = int((screen_width-app_width)/2)
app_y = int((screen_height-app_height)/2-30)

main_window.geometry(f'{app_width}x{app_height}+{app_x}+{app_y}')
main_window.title('Solubility Data Analysis')

lbl_message = tk.Label(main_window, text = "MSSOL")
lbl_message.pack(side = tk.TOP, pady=0)

frm_fpath = tk.Frame(main_window)
txt_fpath = tk.Label(frm_fpath, textvariable=fname, width=fwdth,fg = "black", bg="white", anchor=tk.W)
txt_fpath.grid(row=0, column = 1, padx=5)
btn_fpath = tk.Button(frm_fpath, text = 'Data File (.csv)', width=15, command = lambda:read_input_file())
btn_fpath.grid(row=0,column=0, padx=5)
txt_mpath = tk.Label(frm_fpath, textvariable=mname, width=mwdth,fg = "black", bg="white", anchor=tk.W)
txt_mpath.grid(row=1, column = 1, padx=5)
btn_mpath = tk.Button(frm_fpath, text = 'Metadata File (.csv)', width=15, command = lambda:read_meta_file())
btn_mpath.grid(row=1,column=0, padx=5)
frm_fpath.pack(pady=5)

frm_main = tk.Frame(main_window)

##########################################################################################################

frm_overall = tk.Frame(main_window)

frm_specs = tk.Frame(frm_overall)
frm_specs.grid(row=0, column=0)

frm_data = tk.Frame(frm_specs, bd=2, relief=tk.RIDGE)
frm_data.grid(row=0, column=0)
lbl_data01a = tk.Label(frm_data, textvariable = data01_lbl, width=18).grid(row=0,column=0)
lbl_data01b = tk.Label(frm_data, textvariable = data01_val, width=10).grid(row=0,column=1)
lbl_data02a = tk.Label(frm_data, textvariable = data02_lbl, width=18).grid(row=1,column=0)
lbl_data02b = tk.Label(frm_data, textvariable = data02_val, width=10).grid(row=1,column=1)

frm_meta = tk.Frame(frm_specs, bd=2, relief=tk.RIDGE)
frm_meta.grid(row=1, column=0, padx=12)
lbl_meta01a = tk.Label(frm_meta,textvariable=meta01_lbl, width=18).grid(row=0,column=0)
lbl_meta01b = tk.Label(frm_meta,textvariable=meta01_val, width=10).grid(row=0,column=1)
lbl_meta02a = tk.Label(frm_meta,textvariable=meta02_lbl, width=18).grid(row=1,column=0)
lbl_meta02b = tk.Label(frm_meta,textvariable=meta02_val, width=10).grid(row=1,column=1)
lbl_meta03a = tk.Label(frm_meta,textvariable=meta03_lbl, width=18).grid(row=2,column=0)
lbl_meta03b = tk.Label(frm_meta,textvariable=meta03_val, width=10).grid(row=2,column=1)
lbl_meta04a = tk.Label(frm_meta,textvariable=meta04_lbl, width=18).grid(row=3,column=0)
lbl_meta04b = tk.Label(frm_meta,textvariable=meta04_val, width=10).grid(row=3,column=1)
lbl_meta05a = tk.Label(frm_meta,textvariable=meta05_lbl, width=18).grid(row=4,column=0)
lbl_meta05b = tk.Label(frm_meta,textvariable=meta05_val, width=10).grid(row=4,column=1)
lbl_meta06a = tk.Label(frm_meta,textvariable=meta06_lbl, width=18).grid(row=5,column=0)
lbl_meta06b = tk.Label(frm_meta,textvariable=meta06_val, width=10).grid(row=5,column=1)

frm_graph = tk.Frame(frm_overall)
frm_graph.grid(row=0, column=1, pady=4)
big_plot = tk.Frame(frm_graph)
big_plot.grid(row=0, column=0, pady=4)
f0 = Figure(figsize=(11.8,3), dpi=70, tight_layout=True)
p0 = f0.add_subplot(111)
bigraw_plot = FigureCanvasTkAgg(f0, big_plot)
bigraw_plot.get_tk_widget().grid(row=0, column=1)

frm_slider = tk.Frame(frm_graph)
frm_slider.grid(row=1, column=0, pady=4)
lbl_spacer_1 = tk.Label(frm_slider, text=" ", width=8).grid(row=0, column=0)
# lbl_slider = tk.Label(frm_slider, text=get_current_slider_value(), width=5)
# lbl_slider.grid(row=0, column=1)
slider = tk.Scale(frm_slider,from_=0, variable=val_slider, orient='horizontal', length=760)#, command=slider_changed)
# slider.set(1)
slider['to']=100
slider['tickinterval']=0
slider['showvalue']=0
slider.bind("<ButtonRelease-1>", slider_changed)
slider.grid(row=0, column=2)
#slider.state(['disabled'])
lbl_spacer_2 = tk.Label(frm_slider, text="", width=1).grid(row=0, column=3)

frm_cut = tk.Frame(frm_graph)
frm_cut.grid(row=2, column=0)
lbl_run = tk.Label(frm_cut, text="Run #", anchor=tk.W, justify=tk.LEFT)
lbl_run.grid(row=0, column=0)
ent_run = tk.Spinbox(frm_cut, from_=1, to=5, width=5, fg="black", bg="white", relief=tk.SUNKEN)
ent_run.grid(row=0, column=1, padx=5)


lbl_cut_min = tk.Label(frm_cut, text="Start Time", anchor=tk.W, justify=tk.LEFT)
lbl_cut_min.grid(row=0, column=2)
ent_cut_min = tk.Entry(frm_cut, textvariable= t0_val, width = 10, fg = "black", bg="white", relief=tk.SUNKEN)
ent_cut_min.grid(row=0, column=3, padx=5)
# ent_cut_min.insert(0,"144850")
lbl_cut_max = tk.Label(frm_cut, text="End Time", anchor=tk.W, justify=tk.LEFT)
lbl_cut_max.grid(row=0, column=4)
ent_cut_max = tk.Entry(frm_cut, textvariable= t1_val, width = 10, fg = "black", bg="white", relief=tk.SUNKEN)
ent_cut_max.grid(row=0, column=5, padx=5)
# ent_cut_max.insert(0,"349350")

btn_cut_check = tk.Button(frm_cut, text = 'Check', width=15, command = lambda:cut_data())
btn_cut_check.grid(row=0,column=6, padx=5)

btn_cut_check = tk.Button(frm_cut, text = 'Fit', width=15, command = lambda:fit_analysis())
btn_cut_check.grid(row=0,column=7, padx=5)

frm_overall.pack()


##########################################################################################################
frm_allresults = tk.Frame(main_window)

frm_numeric = tk.Frame(frm_allresults)
frm_numeric.grid(row=0, column=0)

frm_fit = tk.Frame(frm_numeric, bd=2, relief=tk.RIDGE)
frm_fit.grid(row=0, column=0, padx=12)
lbl_fiteq = tk.Label(frm_fit,text="Fit Equation:").grid(row=0,column=0)
lbl_fiteq2 = tk.Label(frm_fit,text="ae^(-bx) + c").grid(row=0,column=1, padx=32)
lbl_fita = tk.Label(frm_fit,text="a =", width=8, anchor=tk.E).grid(row=1,column=0, padx=5)
lbl_fitav = tk.Label(frm_fit,textvariable=fit_params_a, bg="white", width=12, relief=tk.GROOVE).grid(row=1,column=1, padx=15)
lbl_fitb = tk.Label(frm_fit,text="b =", width=8, anchor=tk.E).grid(row=2,column=0, padx=5)
lbl_fitbv = tk.Label(frm_fit,textvariable=fit_params_b, bg="white", width=12, relief=tk.GROOVE).grid(row=2,column=1, padx=15)
lbl_fitc = tk.Label(frm_fit,text="c =", width=8, anchor=tk.E).grid(row=3,column=0, padx=5)
lbl_fitcv = tk.Label(frm_fit,textvariable=fit_params_c, bg="white", width=12, relief=tk.GROOVE).grid(row=3,column=1, padx=15)
lbl_rsq = tk.Label(frm_fit,text="r^2 =", width=8, anchor=tk.E).grid(row=4,column=0, padx=5)
lbl_rsq_r = tk.Label(frm_fit,textvariable=val_rsq, bg="white", width=12, relief=tk.GROOVE).grid(row=4,column=1, padx=15)

frm_details = tk.Frame(frm_numeric, bd=2, relief=tk.RIDGE)
frm_details.grid(row=1, column=0, padx=12)
lbl_det01a = tk.Label(frm_details,textvariable=det01_lbl, width=18, anchor=tk.W, justify=tk.LEFT).grid(row=0,column=0)
lbl_det01b = tk.Label(frm_details,textvariable=det01_val, width=10, anchor=tk.W, justify=tk.LEFT).grid(row=0,column=1)
lbl_det02a = tk.Label(frm_details,textvariable=det02_lbl, width=18, anchor=tk.W, justify=tk.LEFT).grid(row=1,column=0)
lbl_det02b = tk.Label(frm_details,textvariable=det02_val, width=10, anchor=tk.W, justify=tk.LEFT).grid(row=1,column=1)
lbl_det03a = tk.Label(frm_details,textvariable=det03_lbl, width=18, anchor=tk.W, justify=tk.LEFT).grid(row=2,column=0)
lbl_det03b = tk.Label(frm_details,textvariable=det03_val, width=10, anchor=tk.W, justify=tk.LEFT).grid(row=2,column=1)
lbl_det04a = tk.Label(frm_details,textvariable=det04_lbl, width=18, anchor=tk.W, justify=tk.LEFT).grid(row=3,column=0)
lbl_det04b = tk.Label(frm_details,textvariable=det04_val, width=10, anchor=tk.W, justify=tk.LEFT).grid(row=3,column=1)
lbl_det05a = tk.Label(frm_details,textvariable=det05_lbl, width=18, anchor=tk.W, justify=tk.LEFT).grid(row=4,column=0)
lbl_det05b = tk.Label(frm_details,textvariable=det05_val, width=10, anchor=tk.W, justify=tk.LEFT).grid(row=4,column=1)
# lbl_det06a = tk.Label(frm_details,textvariable=det06_lbl, width=13, anchor=tk.W, justify=tk.LEFT).grid(row=5,column=0)
# lbl_det06b = tk.Label(frm_details,textvariable=det06_val, width=10, anchor=tk.W, justify=tk.CENTER).grid(row=5,column=1)

frm_solb =  tk.Frame(frm_numeric, bd=2, relief=tk.RIDGE)
frm_solb.grid(row=2, column=0, padx=12)
lbl_solb01a = tk.Label(frm_solb, textvariable=solb01_lbl, width=18).grid(row=0,column=0)
lbl_solb01b = tk.Label(frm_solb, textvariable=solb01_val, width=10, bg="white", relief=tk.GROOVE).grid(row=0,column=1)
lbl_solb02a = tk.Label(frm_solb, textvariable=solb02_lbl, width=18).grid(row=1,column=0)
lbl_solb02b = tk.Label(frm_solb, textvariable=solb02_val, width=10, bg="white", relief=tk.GROOVE).grid(row=1,column=1)

frm_plot = tk.Frame(frm_allresults)
frm_plot.grid(row=0,column=1)
f1 = Figure(figsize=(5.5,3.2), dpi=75, tight_layout=True)
p1 = f1.add_subplot(111)
raw_plot = FigureCanvasTkAgg(f1, frm_plot)
raw_plot.get_tk_widget().grid(row=0, column=0)
f2 = Figure(figsize=(5.5,3.2), dpi=75, tight_layout=True)
p2 = f2.add_subplot(111)
fit_plot = FigureCanvasTkAgg(f2, frm_plot)
fit_plot.get_tk_widget().grid(row=0, column=1)

frm_allresults.pack()
###########################################################################################################


btn_exit = tk.Button(main_window, text = 'Exit', width=25,command=main_window.destroy)
btn_exit.pack(side=tk.BOTTOM, pady=1)

main_window.mainloop()

 # COLORS
 # 'cornflowerblue'
 # 'turquoise'
 # 'lightseagreen'
 # 'greenyellow'
 # 'olivedrab'
 # 'yellow'
 # 'gold'
 # 'sandybrown'
 # 'orange'
 # 'lightcoral'
 # 'red'