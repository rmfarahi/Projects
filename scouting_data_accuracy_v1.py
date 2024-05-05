# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:26:06 2024

@author: Roxanne Farahi
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfile

np.seterr(divide='ignore', invalid='ignore')

class data_matrix:
    def __init__(self,sc_data,ba_data):
        data_matrix.sc_data = sc_data
        data_matrix.ba_data = ba_data
        # data_matrix.average_acc = average_acc

class dfs:
    def __init__(self,df1,df2,df3):
        dfs.df1 = df1
        dfs.df2 = df2
        dfs.df3 = df3
    def update_df1(df1):   
        dfs.df1 = df1
    def update_df2(df2):   
        dfs.df2 = df2
    def update_df3(df3):   
        dfs.df3 = df3
    

def read_input_file():      
    scouting_file = askopenfile(mode ='r', filetypes=[('csv Files','*.csv'), ('xlsx Files', '*.xlsx')])
    f1name.set(scouting_file.name)
    blue_alliance_file = askopenfile(mode ='r', filetypes=[('csv Files','*.csv'), ('xlsx Files', '*.xlsx')])
    f2name.set(blue_alliance_file.name)

    sc_data = pd.read_csv(scouting_file.name, usecols = ['Match No.','Driver Station Tag','Team No.','Total Points Scored','Scout Name'])
    ba_data = pd.read_csv(blue_alliance_file.name, usecols = ['Match','R_score','B_score'])
    data_matrix(sc_data, ba_data)

def analysis():
    sc_data = data_matrix.sc_data
    ba_data = data_matrix.ba_data
    
    match = sc_data["Match No."]
    dst = sc_data["Driver Station Tag"]
    team = sc_data["Team No."]
    score = sc_data["Total Points Scored"]
    entries = len(match)
    # print(sc_data)
    
    ba_match = ba_data["Match"]
    red_tru = ba_data["R_score"]
    blue_tru = ba_data["B_score"]
    tot_matches = int(ba_match.max())
    # print(ba_data)
    
    # Create empty matrix to map match data
    match_data = []
    for i in range(tot_matches + 1):
        row = []
        for j in range(19):
            row.append(0)
        match_data.append(row)
    
    # Enter match nos., teams, and their corresponding scores into the match_data matrix
    for i in range(entries):
        for j in range(1, tot_matches + 1):
            if match[i] == j and dst[i] == 'R1':
                match_data[j][0] = match[i] 
                match_data[j][1] = team[i]
                match_data[j][2] = score[i]
                
            elif match[i] == j and dst[i] == 'R2':
                match_data[j][0] = match[i] 
                match_data[j][3] = team[i]
                match_data[j][4] = score[i]
     
            elif match[i] == j and dst[i] == 'R3':
                match_data[j][0] = match[i] 
                match_data[j][5] = team[i]
                match_data[j][6] = score[i]
            
            elif match[i] == j and dst[i] == 'B1':
                match_data[j][0] = match[i] 
                match_data[j][10] = team[i]
                match_data[j][11] = score[i]
            
            elif match[i] == j and dst[i] == 'B2':
                match_data[j][0] = match[i] 
                match_data[j][12] = team[i]
                match_data[j][13] = score[i]
          
            elif match[i] == j and dst[i] == 'B3':
                match_data[j][0] = match[i] 
                match_data[j][14] = team[i]
                match_data[j][15] = score[i]
    # print(match_data)
    
    # Enter summed alliance scores, true alliance scores, and calculate scouting accuracy
    for i in range(1, tot_matches + 1):
        red_sum = match_data[i][2] + match_data[i][4] + match_data[i][6]
        match_data[i][7] = red_sum
        match_data[i][8] = red_tru[i] 
        red_acc = 100 - (abs(match_data[i][7] - match_data[i][8]) / match_data[i][8])*100
        match_data[i][9] = red_acc
        
        blue_sum = match_data[i][11] + match_data[i][13] + match_data[i][15]
        match_data[i][16] = blue_sum
        match_data[i][17] = blue_tru[i]
        blue_acc = 100 - (abs(match_data[i][16] - match_data[i][17]) / match_data[i][17])*100
        match_data[i][18] = blue_acc
    
    df1 = pd.DataFrame(match_data, columns=['Match','R1','R1_score','R2','R2_score','R3','R3_score', 
                                           'R_tot','R_tru','R_acc','B1','B1_score','B2','B2_score',
                                           'B3','B3_score', 'B_tot','B_tru','B_acc'])
        
    # Combined matches with offset matches
    comb_match = np.linspace(1,tot_matches,2*tot_matches-1)
    
    # Combined red and blue alliance accuracy
    comb_acc = []
    for i in range(1, tot_matches + 1):
        comb_acc.append(match_data[i][9])
        comb_acc.append(match_data[i][18])
    
    # Average accuracy
    average_acc = sum(comb_acc) / len(comb_acc) 
    val_avg_acc.set("{0:6.1f}".format(average_acc))
    
    # Combined scouted alliance scores
    comb_sc_scores = []
    for i in range(1, tot_matches + 1):
        comb_sc_scores.append(match_data[i][7])
        comb_sc_scores.append(match_data[i][16])
    
    # Combined actual alliance scores
    comb_tru_scores = []
    for i in range(1, tot_matches + 1):
        comb_tru_scores.append(match_data[i][8])
        comb_tru_scores.append(match_data[i][17])
        
    comb1 = pd.DataFrame(comb_match)
    comb2 = pd.DataFrame(comb_acc)
    comb3 = pd.DataFrame(comb_sc_scores)
    comb4 = pd.DataFrame(comb_tru_scores)
    df2 = pd.concat([comb1, comb2, comb3, comb4], axis="columns")
    df2.columns = ['Match','Accuracy','Scouted Score','Actual Score']
        
    name = sc_data["Scout Name"]    
    # Create empty matrix for scout names
    names_data = []
    for i in range(tot_matches):
        row = []
        for j in range(7):
            row.append(0)
        names_data.append(row)
    
    # Enter match nos., teams, and their corresponding scores into the match_data matrix
    for i in range(entries):
        for j in range(1, tot_matches + 1):
            if match[i] == j and dst[i] == 'R1':
                names_data[j][0] = match[i] 
                names_data[j][1] = name[i]
            elif match[i] == j and dst[i] == 'R2':
                names_data[j][0] = match[i] 
                names_data[j][2] = name[i]
            elif match[i] == j and dst[i] == 'R3':
                names_data[j][0] = match[i] 
                names_data[j][3] = name[i]
            elif match[i] == j and dst[i] == 'B1':
                names_data[j][0] = match[i] 
                names_data[j][4] = name[i]
            elif match[i] == j and dst[i] == 'B2':
                names_data[j][0] = match[i] 
                names_data[j][5] = name[i]
            elif match[i] == j and dst[i] == 'B3':
                names_data[j][0] = match[i] 
                names_data[j][6] = name[i]
    df3 = pd.DataFrame(names_data, columns=['Match','R1','R2','R3','B1','B2','B3'])
    dfs(df1,df2,df3)
    
    
def save_dfs():
    df1 = dfs.df1
    df2 = dfs.df2
    df3 = dfs.df3
    df1.to_csv('SC-BA_Match_Data.csv', index = False)
    df2.to_csv('Merged_Alli_Data.csv', index = False)
    df3.to_csv('Scout_Names_Data.csv', index = False)

main_window = tk.Tk()
f1name= tk.StringVar()
f2name = tk.StringVar()
val_avg_acc = tk.StringVar()

main_window.geometry('750x230')
# Create frame for file input
frm_fpath = tk.Frame(main_window)

# Input file button
btn_fpath = tk.Button(frm_fpath, text = 'Input File (.csv)', width=15, command = lambda:read_input_file())
btn_fpath.grid(row=0,column=0, padx=5)

# Text specifying scouting data input
lbl_fpath = tk.Label(frm_fpath, text="Scouting", width=10, anchor=tk.W).grid(row=1, column = 0, padx=5, pady=7)
txt_fpath = tk.Label(frm_fpath, textvariable=f1name, width=80,fg = "black", bg="white", anchor=tk.W)
txt_fpath.grid(row=1, column = 1, padx=5, pady=5)

# Text specifying blue alliance data input
lbl_fpath = tk.Label(frm_fpath, text="Blue Alliance", width=10, anchor=tk.W).grid(row=2, column = 0, padx=5, pady=7)
txt_fpath2 = tk.Label(frm_fpath, textvariable=f2name, width=80,fg = "black", bg="white", anchor=tk.W)
txt_fpath2.grid(row=2, column = 1, padx=5, pady=5)
frm_fpath.pack()


# Create frame for accuracy
frm_analysis = tk.Frame(main_window)
# Perform Analysis button
btn_analysis = tk.Button(frm_analysis, text = 'Analyze Data', width=15, command = lambda:analysis())
btn_analysis.grid(row=0,column=0, padx=5, pady=5)
# Export data button
lbl_analysis = tk.Label(frm_analysis,text="Average Accuracy %").grid(row=0,column=1, padx=5)
lbl_analysis_r = tk.Label(frm_analysis,textvariable=val_avg_acc, width=8,fg = "black", bg="white", anchor=tk.W).grid(row=0,column=2, padx=5, pady=5)
frm_analysis.pack()

# Create fram to save data
frm_save = tk.Frame(main_window)
btn_save = tk.Button(frm_save, text = 'Export Analysis', width=15, command = lambda:save_dfs())
btn_save.grid(row=0,column=1, pady=7)
frm_save.pack()

btn_exit = tk.Button(main_window, text = 'Exit', width=25,command=main_window.destroy)
btn_exit.pack(side=tk.BOTTOM, pady=10)

main_window.mainloop()

