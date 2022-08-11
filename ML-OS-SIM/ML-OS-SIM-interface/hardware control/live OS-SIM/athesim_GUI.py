from datetime import datetime
from tkinter.constants import DISABLED, HORIZONTAL, NORMAL
from PIL import Image, ImageTk # numpy to GUI element
import tkinter as tk
from tkinter import TRUE, ttk
import threading
import torch.multiprocessing as mp
import numpy as np
import athesim_functions as asf
import time
from pycromanager import Bridge # camera control
import torch
from tkinter.messagebox import showinfo
import serial

asi_stage_serialport = 'COM3'
asi_stage_baudrate = 115200

## ML-SIM App
class ML_App:
    def __init__(self,master):

        self.master = master
        master.title('Live OS-SIM')
        master.geometry("700x640") # size of gui
        tabControl = ttk.Notebook(self.master)
        
        self.tab1 = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text ='Acquisition control')
        tabControl.add(self.tab2, text ='Hardware properties')
        tabControl.place(x = 5,y = 60, width = 690, height = 585)

        self.stop_signal = mp.Queue()
        self.output = mp.Queue()
        self.stack = mp.Queue()

        # Use optosplit
        self.opto = tk.IntVar()
        self.multi = tk.Checkbutton(self.tab2,variable=self.opto)
        self.multi.place(x=15, y=120)
        # Acquire Red channel
        self.R = tk.IntVar()
        self.rChan = tk.Checkbutton(self.tab2,variable=self.R)
        self.rChan.place(x=118, y=278)
        # Acquire Green channel
        self.G = tk.IntVar()
        self.gChan = tk.Checkbutton(self.tab2,variable=self.G)
        self.gChan.place(x=118, y=301)
        # Acquire Blue channel
        self.B = tk.IntVar()
        self.bChan = tk.Checkbutton(self.tab2,variable=self.B)
        self.bChan.place(x=118, y=324)

        self.multi_label = tk.Label(self.tab2, text = "Use optosplit")
        self.multi_label.place(x = 30,y = 122)

        self.y1 = tk.IntVar()
        self.y1.set(700)
        self.yco1 = tk.Entry(self.tab2,text='Y1',textvariable=self.y1) # Y1 field
        self.yco1.place(x=75, y=280, width=25)
        self.y1_label = tk.Label(self.tab2, text = "y1")
        self.y1_label.place(x = 100,y = 280)

        self.x1 = tk.IntVar()
        self.x1.set(22)
        self.xco1 = tk.Entry(self.tab2,textvariable=self.x1) # X1 field
        self.xco1.place(x=15, y=280, width=25)
        self.x1_label = tk.Label(self.tab2, text = "x1")
        self.x1_label.place(x = 40,y = 280)        

        self.y2 = tk.IntVar()
        self.y2.set(700)
        self.yco2 = tk.Entry(self.tab2,textvariable=self.y2) # Y2 field
        self.yco2.place(x=75, y=303, width=25)
        self.y2_label = tk.Label(self.tab2, text = "y2")
        self.y2_label.place(x = 100,y=303)

        self.x2 = tk.IntVar()
        self.x2.set(695)
        self.xco2 = tk.Entry(self.tab2,textvariable=self.x2) # X2 field
        self.xco2.place(x=15, y=303, width=25)
        self.x2_label = tk.Label(self.tab2, text = "x2")
        self.x2_label.place(x = 40,y=303)  

        self.y3 = tk.IntVar()
        self.y3.set(700)
        self.yco3 = tk.Entry(self.tab2,textvariable=self.y3) # Y3 field
        self.yco3.place(x=75, y=326, width=25)
        self.y3_label = tk.Label(self.tab2, text = "y3")
        self.y3_label.place(x = 100,y=326)  

        self.x3 = tk.IntVar()
        self.x3 .set(1385)
        self.xco3 = tk.Entry(self.tab2,textvariable=self.x3) # X3 field
        self.xco3.place(x=15, y=326, width=25)
        self.x3_label = tk.Label(self.tab2, text = "x3 ")
        self.x3_label.place(x = 40,y=326) 

        self.opto_text = tk.Label(self.tab2, text = "Optosplit parameters")
        self.opto_text.place(x = 15,y=257)         

        
        self.live = tk.Button(self.tab1, width=10, text='Preview', command = self.start_live)
        self.live.place(x=15, y=20)

        self.Stop_live = tk.Button(self.tab1, width=10, text='Stop', command = self.stop_live)
        self.Stop_live.place(x=15, y=50)

        blank = np.zeros((512,512))
        blank = blank.astype('uint8')
        img =  ImageTk.PhotoImage(image=Image.fromarray(blank)) # image
        self.panel = tk.Label(self.tab1, image=img)
        self.panel.configure(image=img) # update the GUI element
        self.panel.image = img  
        self.panel.place(x=155, y=20)

        imgo = Image.open('C:/Users/SIM_Admin/Documents/GitHub/ML-OS-SIM/ML-OS-SIM-interface/hardware control/live OS-SIM/optosplit.jpg')
        test =  ImageTk.PhotoImage(imgo)
        self.optosplit = tk.Label(self.tab2, image=test)
        self.optosplit.image = test  
        self.optosplit.place(x=150, y=20)

        imgo = Image.open('C:/Users/SIM_Admin/Documents/GitHub/ML-OS-SIM/ML-OS-SIM-interface/hardware control/live OS-SIM/Clipboard.png')
        test =  ImageTk.PhotoImage(imgo)
        self.logo = tk.Label(image=test)
        self.logo.image = test  
        self.logo.place(x=3, y=3)


        self.quit_button = tk.Button(self.tab1,width=10, fg = "red", text='Quit',command=self.quit_gui) # quit the GUI
        self.quit_button.place(x=15, y=170)    

        # Testing Z button for stage control 
        self.z_button = tk.Button(self.tab1, width=10, text='Z Axis-OS', command = self.z_control)
        self.z_button.place(x=15, y=525)

        self.laservariable=0
        self.laser2variable=0
        self.laser3variable=0   

        ## laser control gui
        self.laser1button_On = tk.Button(self.tab1, width=8, text='L561 On', command = self.laser_control_checkbox)
        self.laser1button_On.place(x=10, y=210)

        self.laser1button_Off = tk.Button(self.tab1, width=8, text='L561 Off', command = self.laser_control_checkbox_off)
        self.laser1button_Off.place(x=80, y=210)

        self.l561_label = tk.Label(self.tab1, text = "L561 Power (mW)")
        self.l561_label.place(x = 11,y = 240)

        l561_power_range = list(range(1,101))
        l561_selected_power = tk.StringVar()
        l561_power_cb = ttk.Combobox(self.tab1, textvariable=l561_selected_power)
        l561_power_cb['values'] = [l561_power_range[p] for p in range(0,100)]
        l561_power_cb['state'] = 'readonly'
        l561_power_cb.place(x=115,y=240,width=35)

        def l561_power_changed(event):
            showinfo( 
                title='Result', message=f'You selected {l561_selected_power.get()} mW!'
                )
            asf.laserAOTF_power_control(self.laservariable,l561_selected_power.get())
        
        l561_power_cb.bind('<<ComboboxSelected>>', l561_power_changed)

        self.laser2button_On = tk.Button(self.tab1, width=8, text='L488 On', command = self.laser2_control_checkbox)
        self.laser2button_On.place(x=10, y=270)

        self.laser2button_Off = tk.Button(self.tab1, width=8, text='L488 Off', command = self.laser2_control_checkbox_off)
        self.laser2button_Off.place(x=80, y=270)

        self.l488_label = tk.Label(self.tab1, text = "L488 Power (mW)")
        self.l488_label.place(x = 11,y = 300)

        l488_power_range = list(range(1,101))
        l488_selected_power = tk.StringVar()
        l488_power_cb = ttk.Combobox(self.tab1, textvariable=l488_selected_power)
        l488_power_cb['values'] = [l488_power_range[p] for p in range(0,100)]
        l488_power_cb['state'] = 'readonly'
        l488_power_cb.place(x=115,y=300,width=35)

        def l488_power_changed(event):
            showinfo( 
                title='Result', message=f'You selected {l488_selected_power.get()} mW!'
                )
            asf.laser2_power_control(self.laser2variable,l488_selected_power.get())
        
        l488_power_cb.bind('<<ComboboxSelected>>', l488_power_changed)


        self.laser3button_On = tk.Button(self.tab1, width=8, text='L405 On', command = self.laser3_control_checkbox)
        self.laser3button_On.place(x=10, y=330)

        self.laser3button_Off = tk.Button(self.tab1, width=8, text='L405 Off', command = self.laser3_control_checkbox_off)
        self.laser3button_Off.place(x=80, y=330)

        self.l405_label = tk.Label(self.tab1, text = "L405 Power (mW)")
        self.l405_label.place(x = 11,y = 360)

        l405_power_range = list(range(1,101))
        l405_selected_power = tk.StringVar()
        l405_power_cb = ttk.Combobox(self.tab1, textvariable=l405_selected_power)
        l405_power_cb['values'] = [l405_power_range[p] for p in range(0,100)]
        l405_power_cb['state'] = 'readonly'
        l405_power_cb.place(x=115,y=360,width=35)

        def l405_power_changed(event):
            showinfo( 
                title='Result', message=f'You selected {l405_selected_power.get()} mW!'
                )
            asf.laser3_power_control(self.laser3variable,l405_selected_power.get())
        
        l405_power_cb.bind('<<ComboboxSelected>>', l405_power_changed)

        #647 Laser

        self.laser4button_On = tk.Button(self.tab1, width=8, text='L647 On', command = self.laser4_control_checkbox)
        self.laser4button_On.place(x=10, y=390)

        self.laser4button_Off = tk.Button(self.tab1, width=8, text='L647 Off', command = self.laser4_control_checkbox_off)
        self.laser4button_Off.place(x=80, y=390)

        self.l647_label = tk.Label(self.tab1, text = "L647 Power (mW)")
        self.l647_label.place(x = 11,y = 420)

        l647_power_range = list(range(1,101))
        l647_selected_power = tk.StringVar()
        l647_power_cb = ttk.Combobox(self.tab1, textvariable=l647_selected_power)
        l647_power_cb['values'] = [l647_power_range[p] for p in range(0,100)]
        l647_power_cb['state'] = 'readonly'
        l647_power_cb.place(x=115,y=420,width=35)

        def l647_power_changed(event):
            showinfo( 
                title='Result', message=f'You selected {l647_selected_power.get()}!'
                )
            asf.laser4_power_control(self.laser4variable,l647_selected_power.get())
        
        l647_power_cb.bind('<<ComboboxSelected>>', l647_power_changed)


        self.start_live_decon = tk.Button(self.tab1,width=10, text='Live OS-SIM', command = self.start_os_sim) # start live sim
        self.start_live_decon.place(x=15, y=80)

        self.update_ROI = tk.Button(self.tab2,width=10, text='Update ROI') #update camera ROI
        self.update_ROI.place(x=15, y=220)

        self.display_label = tk.Label(self.tab1, text = "Display range")
        self.display_label.place(x = 13,y = 440)

        self.iMin = tk.IntVar()
        self.iMin.set(0)
        self.limLow = tk.Entry(self.tab1,textvariable=self.iMin) # Display range field
        self.limLow.place(x=15, y=460, width=30)

        self.iMax = tk.IntVar()
        self.iMax.set(100)
        self.limHigh = tk.Entry(self.tab1,textvariable=self.iMax) # Display range field
        self.limHigh.place(x=50, y=460, width=30)

        self.display_label = tk.Label(self.tab1, text = "Reconstruction range")
        self.display_label.place(x = 13,y = 480)

        self.rMin = tk.IntVar()
        self.rMin.set(50)
        self.rlimLow = tk.Entry(self.tab1,textvariable=self.rMin) # Reconstruction range field
        self.rlimLow.place(x=15, y=500, width=30)

        self.rMax = tk.IntVar()
        self.rMax.set(1000)
        self.rlimHigh = tk.Entry(self.tab1,textvariable=self.rMax) # Reconstruction range field
        self.rlimHigh.place(x=50, y=500, width=30)

        self.save_image_btn = tk.Button(self.tab1,width=10, text='Save Image', command = self.save_image) # capture and save current image
        self.save_image_btn.place(x=15, y=110)

        self.exposure_label = tk.Label(self.tab1, text = "Exposure time (ms)")
        self.exposure_label.place(x = 13,y = 143)

        self.expTime = tk.IntVar()
        self.expTime.set(80)
        self.exposure = tk.Entry(self.tab1,textvariable=self.expTime) # exposure time field
        self.exposure.place(x=120, y=143, width=35)

        self.xOff = tk.IntVar()
        self.xOff.set(710)
        self.xoffset = tk.Entry(self.tab2,textvariable=self.xOff) # ROI input
        self.xoffset.place(x=20, y=174, width=50)
        self.xoffset_label = tk.Label(self.tab2, text = "ROI offset")
        self.xoffset_label.place(x = 15,y = 154)

        self.yOff = tk.IntVar()
        self.yOff.set(700)
        self.yoffset = tk.Entry(self.tab2,textvariable=self.yOff) # ROI input
        self.yoffset.place(x=20, y=195, width=50)
        
        if not torch.cuda.is_available():
            self.start_live_decon['state'] = DISABLED
            print('A valid GPU is required for live OS-SIM')
        else:
            gpu_dev = torch.cuda.get_device_name(0)
            print('Using device:')
            print(gpu_dev)
    
    ## Class functions
    def update_roi(self):
        xOffset = self.xOff.get() # get ROI variables from the GUI input
        yOffset = self.yOff.get()
        if xOffset < 1500 and yOffset < 1500: # make sure ROI is valid
            self.stop_live()
            time.sleep(0.1) #wait for other processes to stop
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                ROI = [xOffset, yOffset, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI    
    
    def laser_control_checkbox(self):
        self.laservariable = 1
        asf.laser_control(self.laservariable)
        self.laser1button_On['state'] = tk.DISABLED
        self.laser1button_Off['state'] = tk.NORMAL

    def laser_control_checkbox_off(self):
        self.laservariable = 0
        asf.laser_control(self.laservariable)
        self.laser1button_Off['state'] = tk.DISABLED
        self.laser1button_On['state'] = tk.NORMAL

    def laser2_control_checkbox(self):
        self.laser2variable = 1
        asf.laser2_control(self.laser2variable)
        self.laser2button_On['state'] = tk.DISABLED
        self.laser2button_Off['state'] = tk.NORMAL

    def laser2_control_checkbox_off(self):
        self.laser2variable = 0
        asf.laser2_control(self.laser2variable)
        self.laser2button_Off['state'] = tk.DISABLED
        self.laser2button_On['state'] = tk.NORMAL
    
    def laser3_control_checkbox(self):
        self.laser3variable = 1
        asf.laser3_control(self.laser3variable)
        self.laser3button_On['state'] = tk.DISABLED
        self.laser3button_Off['state'] = tk.NORMAL

    def laser3_control_checkbox_off(self):
        self.laser3variable = 0
        asf.laser3_control(self.laser3variable)
        self.laser3button_Off['state'] = tk.DISABLED
        self.laser3button_On['state'] = tk.NORMAL
    
    def laser4_control_checkbox(self):
        self.laser4variable = 1
        asf.laser4_control(self.laser4variable)
        self.laser4button_On['state'] = tk.DISABLED
        self.laser4button_Off['state'] = tk.NORMAL

    def laser4_control_checkbox_off(self):
        self.laser4variable = 0
        asf.laser4_control(self.laser4variable)
        self.laser4button_Off['state'] = tk.DISABLED
        self.laser4button_On['state'] = tk.NORMAL

    def start_live(self):
        self.start_live_decon["state"] == DISABLED
        self.quit_button["state"] == DISABLED
        optosplit = self.opto.get()
        R = self.R.get(); G = self.G.get(); B = self.B.get()
        if optosplit == 1:
            x1 = self.x1.get() # get ROI variables from the GUI input
            y1 = self.y1.get()
            x2 = self.x2.get() # get ROI variables from the GUI input
            y2 = self.y2.get()
            x3 = self.x3.get() # get ROI variables from the GUI input
            y3 = self.y3.get()
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                if core.is_sequence_running():
                    core.stop_sequence_acquisition() # stop the camera
                xmin = min(x1,x2,x3)
                xmax = max(x1,x2,x3)
                x1 = x1-xmin; x2 = x2-xmin; x3 = x3-xmin
                width = xmax-xmin+513
                ymin = min(y1,y2,y3)
                ymax = max(y1,y2,y3)
                y1 = y1-ymin; y2 = y2-ymin; y3 = y3-ymin               
                height = ymax-ymin+513
                ROI = [xmin, ymin, width, height] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI for optosplit')
        else:
            with Bridge() as bridge: # load camera control library
                x1 = self.xOff.get() # get ROI variables from the GUI input
                y1 = self.yOff.get()
                x2 = 0
                y2 = 0
                x3 = 0
                y3 = 0                
                core = bridge.get_core()
                ROI = [x1, y1, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI') 


        exposure_time = self.expTime.get()
        child_max = mp.Value('d',1000)
        child_min = mp.Value('d',1)
        rchild_max = mp.Value('d',1000)
        rchild_min = mp.Value('d',1)
        #self.live_process = mp.Process(target= asf.live_view, args = (self.stop_signal,
        #    self.output,exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
        
        #updated live view v2
        self.live_process = mp.Process(target= asf.live_view_v2, args = (self.stop_signal,
            self.output,exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))

        self.live_process.start()
        self.plotting_process = threading.Thread(target= self.plot, args = (child_max,child_min,rchild_max,rchild_min))
        self.plotting_process.start()

    def start_os_sim(self):
        self.live["state"] == DISABLED
        self.quit_button["state"] == DISABLED
        optosplit = self.opto.get()
        R = self.R.get(); G = self.G.get(); B = self.B.get()
        if optosplit == 1:
            x1 = self.x1.get() # get ROI variables from the GUI input
            y1 = self.y1.get()
            x2 = self.x2.get() # get ROI variables from the GUI input
            y2 = self.y2.get()
            x3 = self.x3.get() # get ROI variables from the GUI input
            y3 = self.y3.get()
            with Bridge() as bridge: # load camera control library
                core = bridge.get_core()
                if core.is_sequence_running():
                    core.stop_sequence_acquisition() # stop the camera 
                xmin = min(x1,x2,x3)
                xmax = max(x1,x2,x3)
                x1 = x1-xmin; x2 = x2-xmin; x3 = x3-xmin
                width = xmax-xmin+513
                ymin = min(y1,y2,y3)
                ymax = max(y1,y2,y3)
                y1 = y1-ymin; y2 = y2-ymin; y3 = y3-ymin
                height = ymax-ymin+513
                ROI = [xmin, ymin, width, height] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI for optosplit')
        else:
            with Bridge() as bridge: # load camera control library
                x1 = self.xOff.get() # get ROI variables from the GUI input
                y1 = self.yOff.get()
                x2 = 0
                y2 = 0
                x3 = 0
                y3 = 0
                core = bridge.get_core()
                ROI = [x1, y1, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI')

        exposure_time = self.expTime.get()
        child_max = mp.Value('d',1000)
        child_min = mp.Value('d',1)
        rchild_max = mp.Value('d',1000)
        rchild_min = mp.Value('d',1)

        #self.live_process = mp.Process(target= asf.live_ml_sim, args = (self.stack,self.stop_signal,self.output,
        #    exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))
        
        #updated live os process v2
        self.live_process = mp.Process(target= asf.live_os_sim_v2, args = (self.stack,self.stop_signal,self.output,
            exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))

        self.live_process.start()

        self.plotting_process = threading.Thread(target= self.plot, args = (child_max,child_min,rchild_max,rchild_min))
        self.plotting_process.start()    

    def quit_gui(self):
        self.stop_signal.put(False)
        time.sleep(1)
        self.master.destroy()

    def z_control(self):
        z_min_value = 62.9 # to get to the lowest membrane layer of the cell
        z_max_value = 1030.0 # to get to the surface - uppermost layer of the cell
        stepsize = 96.7 # Steps to get 10 slices 
        # cell z values - Top : 0.00629 bottom : 0.1030 /0.1011  mid = 0.0916 --> 97,98,91,89

        # Communication setup
        ASI_StageSer = serial.Serial(asi_stage_serialport,asi_stage_baudrate)
        ASI_StageSer.bytesize = serial.EIGHTBITS # bits per byte
        ASI_StageSer.parity = serial.PARITY_NONE
        ASI_StageSer.stopbits = serial.STOPBITS_ONE
        ASI_StageSer.timeout = 5
        ASI_StageSer.xonxoff = False #disable software flow conrol
        ASI_StageSer.rtscts = False #disable hardware (RTS/CTS) flow control
        ASI_StageSer.dsrdtr =  False #disable hardware (DSR/DTR) flow control
        ASI_StageSer.writeTimeout = 0 #timeout for write

        print('Starting RS-232 Communication Setup for ASI Stage Control')

        for z in np.arange(z_min_value, z_max_value, stepsize):

            z_axis_value = float(int(z))

            if self.z_button['state']:
                try:
                    ASI_StageSer.open()
                except Exception as e:
                    print('Exception: Opening serial port: '+ str(e))
    
                if ASI_StageSer.isOpen():
                    ASI_ZCommand = 'M Z=' + str(z_axis_value) +'\r\n'
                    ASI_StageSer.write(str.encode(ASI_ZCommand)) 
                    self.start_live()
                    #time.sleep(60)

                    optosplit = self.opto.get()
                    R = self.R.get(); G = self.G.get(); B = self.B.get()
                    
                    with Bridge() as bridge: # load camera control library
                        x1 = self.xOff.get() # get ROI variables from the GUI input
                        y1 = self.yOff.get()
                        x2 = 0
                        y2 = 0
                        x3 = 0
                        y3 = 0                
                        core = bridge.get_core()
                        ROI = [x1, y1, 512, 512] # build ROI 
                        core.set_roi(*ROI) # set ROI  
                        print('Successfully set ROI') 
                    
                    exposure_time = self.expTime.get()
                    child_max = mp.Value('d',1000)
                    child_min = mp.Value('d',1)
                    rchild_max = mp.Value('d',1000)
                    rchild_min = mp.Value('d',1)

                    #updated live view v2
                    self.live_process = mp.Process(target= asf.live_view_v2, args = (self.stop_signal,
                    self.output,exposure_time,optosplit,x1,y1,x2,y2,x3,y3,rchild_max,rchild_min,R,G,B))

                    self.live_process.start()
                    self.plotting_process = threading.Thread(target= self.plot, args = (child_max,child_min,rchild_max,rchild_min))
                    self.plotting_process.start()

                    print("AZI Z Command" + ASI_ZCommand)
                    try:
                        ASI_StageResponse = ASI_StageSer.readline().decode('ascii')
                        print("response data: " +ASI_StageResponse)
                        ASI_StageSer.flush()
                    except Exception as e:
                        print('Exception: Writing to serial port: '+ str(e))
                    print('ASI Stage Z axis moved')
                    self.stop_live()
                    time.sleep(10)
                else:
                    print('Connection failure')
            else:
                try:
                    ASI_StageSer.open()
                except Exception as e:
                    print('Exception: Opening serial port: '+ str(e))

                if ASI_StageSer.isOpen():
                    try:
                        ASI_StageResponse = ASI_StageSer.readline().decode('ascii')
                        print("response data: " +ASI_StageResponse)

                        showinfo( 
                        title='Warning', message=f'AZI Stage Control is Off - Turn On Device for Control!'
                        )
                        ASI_StageSer.flush()
                    except Exception as e:
                        print('Exception: Writing to serial port: '+ str(e))
                    print('ASI Stage control off')
                else:
                    print('Connection failure')

    def stop_live(self):
        self.stop_signal.put(False)
        self.start_live_decon["state"] == NORMAL
        self.live["state"] == NORMAL
        self.quit_button["state"] == NORMAL

    def save_image(self):
        print('Save Image Enabled')
        self.save_image_btn["state"] = DISABLED

    def plot(self,child_max,child_min,rchild_max,rchild_min):
        while True: 

            rMin = self.rMin.get()
            rchild_min.value = rMin
            rMax = self.rMax.get()
            rchild_max.value = rMax

            iMax = self.iMax.get()
            child_max.value = iMax
            iMin = self.iMin.get()
            child_min.value = iMin

            if not self.output.empty():
                image_array = self.output.get() # empty data from reconstruction pool
                if isinstance(image_array, bool):
                    print('finished acquisition')
                    break
                elif len(image_array.shape)==2:
                    # run the update function
                    image_array = image_array-np.amin(image_array)
                    image_array = image_array*(255/np.amax(image_array)) 
                    image_array = image_array.astype('uint8')
                    img =  ImageTk.PhotoImage(image=Image.fromarray(image_array,mode='L')) # convert numpy array to tikner object 
                    self.panel.configure(image=img) # update the GUI element
                    self.panel.image = img  

                    # Capture image and save with appropriate label
                    if(self.save_image_btn["state"] == DISABLED):
                        now = datetime.now()
                        hr = now.hour
                        min = now.minute
                        sec = now.second
                        imageName = 'OS-SIM_Image_'
                        time_label = '_' + str(hr) +'-'+ str(min) + '-' + str(sec)
                        if(self.live["state"] == DISABLED and self.quit_button["state"] == DISABLED):
                            imageName = imageName + 'Recons_'
                        else:
                            imageName = 'OS-SIM_Image_'
                        filename = imageName + str(datetime.now().date()) + time_label + '.png'
                        filepath = 'C:/Users/SIM_Admin/Documents/GitHub/ML-OS-SIM/ML-OS-SIM-interface/hardware control/live OS-SIM/Saved Images/' + filename
                        picture = Image.fromarray(image_array,mode='L')
                        picture = picture.convert('L')
                        picture = picture.save(filepath)
                        print('Image Captured:' + filename)
                        print('Save button status',self.save_image_btn["state"])
                        self.save_image_btn["state"] = NORMAL
                        print('Save button status',self.save_image_btn["state"])
                        self.save_image_btn["state"] = tk.NORMAL
                    else:
                        #print('Save buttton disabled')
                        self.save_image_btn["state"] == NORMAL
                        self.save_image_btn["state"] == tk.NORMAL

                elif len(image_array.shape)==3:
                    r = image_array[:,:,0]
                    g = image_array[:,:,1]
                    result = np.zeros((512,512,3))
                    r = r-np.amin(r)
                    r = 255*(r/np.amax(r))
                    g = g-np.amin(g)
                    g = 255*(g/np.amax(g))
                    result[:,:,0] = r
                    result[:,:,1] = g 
                    result = result.astype('uint8')

                    # Capture image and save with appropriate label
                    if(self.save_image_btn["state"] == DISABLED):
                        now = datetime.now()
                        hr = now.hour
                        min = now.minute
                        sec = now.second
                        imageName = 'OS-SIM_Image_'
                        time_label = '_' + str(hr) +'-'+ str(min) + '-' + str(sec)
                        if(self.live["state"] == DISABLED and self.quit_button["state"] == DISABLED):
                            imageName = imageName + 'Recons_'
                        else:
                            imageName = 'OS-SIM_Image_'
                        filename = imageName + str(datetime.now().date()) + time_label + '.png'
                        filepath = 'C:/Users/SIM_Admin/Documents/GitHub/ML-OS-SIM/ML-OS-SIM-interface/hardware control/live OS-SIM/Saved Images/' + filename
                        picture = Image.fromarray(image_array,mode='L')
                        picture = picture.convert('L')
                        picture = picture.save(filepath)
                        print('Image Captured:' + filename)
                        print('Save button status',self.save_image_btn["state"])
                        self.save_image_btn["state"] = NORMAL
                        print('Save button status',self.save_image_btn["state"])
                        self.save_image_btn["state"] = tk.NORMAL
                    else:
                        #print('Save buttton disabled')
                        self.save_image_btn["state"] == NORMAL
                        self.save_image_btn["state"] == tk.NORMAL
                    

            # else:
                # print('imArray was empty')


if __name__ == '__main__':
    root = tk.Tk()
    my_gui = ML_App(root)
    root.mainloop()
