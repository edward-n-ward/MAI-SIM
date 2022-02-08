from tkinter.constants import DISABLED, HORIZONTAL, NORMAL
from PIL import Image, ImageTk # numpy to GUI element
import tkinter as tk
from tkinter import ttk
import threading
import torch.multiprocessing as mp
import numpy as np
import athesim_functions as asf
import time
from pycromanager import Bridge # camera control
import torch

## ML-SIM App
class ML_App:
    def __init__(self,master):

        self.master = master
        tabControl = ttk.Notebook(self.master)
        master.geometry("780x600") # size of gui
        self.tab1 = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text ='Acquisition control')
        tabControl.add(self.tab2, text ='Hardware properties')
        tabControl.pack(expand = 1, fill ="both")

        self.stop_signal = mp.Queue()
        self.output = mp.Queue()
        self.stack = mp.Queue()

        self.opto = tk.IntVar()
        self.multi = tk.Checkbutton(self.tab2,textvariable=self.opto)
        self.multi.place(x=15, y=120)
        self.multi_label = tk.Label(self.tab2, text = "Use optosplit")
        self.multi_label.place(x = 30,y = 122)

        self.y1 = tk.IntVar()
        self.y1.set(30)
        self.yco1 = tk.Entry(self.tab2,text='Y1',textvariable=self.y1) # Y1 field
        self.yco1.place(x=75, y=280, width=25)
        self.y1_label = tk.Label(self.tab2, text = "y1")
        self.y1_label.place(x = 100,y = 280)

        self.x1 = tk.IntVar()
        self.x1.set(30)
        self.xco1 = tk.Entry(self.tab2,textvariable=self.x1) # X1 field
        self.xco1.place(x=15, y=280, width=25)
        self.x1_label = tk.Label(self.tab2, text = "x1")
        self.x1_label.place(x = 40,y = 280)        

        self.y2 = tk.IntVar()
        self.y2.set(30)
        self.yco2 = tk.Entry(self.tab2,textvariable=self.y2) # Y2 field
        self.yco2.place(x=75, y=303, width=25)
        self.y2_label = tk.Label(self.tab2, text = "y2")
        self.y2_label.place(x = 100,y=303)

        self.x2 = tk.IntVar()
        self.x2.set(30)
        self.xco2 = tk.Entry(self.tab2,textvariable=self.x2) # X2 field
        self.xco2.place(x=15, y=303, width=25)
        self.x2_label = tk.Label(self.tab2, text = "x2")
        self.x2_label.place(x = 40,y=303)  
        self.opto_text = tk.Label(self.tab2, text = "Optosplit parameters")
        self.opto_text.place(x = 15,y=257)         

        
        self.live = tk.Button(self.tab1, width=10, text='Start', command = self.start_live)
        self.live.place(x=15, y=10)
        self.Stop_live = tk.Button(self.tab1, width=10, text='Stop', command = self.stop_live)
        self.Stop_live.place(x=15, y=40)

        blank = np.zeros((512,512))
        blank = blank.astype('uint8')
        img =  ImageTk.PhotoImage(image=Image.fromarray(blank)) # image
        self.panel = tk.Label(self.tab1, image=img)
        self.panel.configure(image=img) # update the GUI element
        self.panel.image = img  
        self.panel.pack(side = "top")

        imgo = Image.open('C:/Users/SIM_ADMIN/Documents/GitHub/AtheSIM/Code/Hardware control/live ML-SIM/optosplit.jpg')
        test =  ImageTk.PhotoImage(imgo)
        self.optosplit = tk.Label(self.tab2, image=test)
        self.optosplit.image = test  
        self.optosplit.pack(side = "top")

        self.live_decon = tk.Button(self.tab1,width=10, text='Live ML-SIM', command = self.start_live) # start live preview
        self.live_decon.place(x=15, y=250)

        self.quit_button = tk.Button(self.tab1,width=10, text='Quit',command=self.quit_gui) # quit the GUI
        self.quit_button.place(x=15, y=250)       

        self.start_live_decon = tk.Button(self.tab1,width=10, text='Live ML-SIM', command = self.start_ml_sim) # start live sim
        self.start_live_decon.place(x=15, y=280)

        self.update_ROI = tk.Button(self.tab2,width=10, text='Update ROI') #update camera ROI
        self.update_ROI.place(x=15, y=220)

        self.expTime = tk.IntVar()
        self.expTime.set(30)
        self.exposure = tk.Entry(self.tab1,textvariable=self.expTime) # exposure time field
        self.exposure.place(x=20, y=130, width=50)

        self.exposure_label = tk.Label(self.tab1, text = "Exposure time (ms)")
        self.exposure_label.place(x = 15,y = 110)

        self.xOff = tk.IntVar()
        self.xOff.set(30)
        self.xoffset = tk.Entry(self.tab2,textvariable=self.xOff) # ROI input
        self.xoffset.place(x=20, y=174, width=50)
        self.xoffset_label = tk.Label(self.tab2, text = "ROI offset")
        self.xoffset_label.place(x = 15,y = 154)

        self.yOff = tk.IntVar()
        self.yOff.set(30)
        self.yoffset = tk.Entry(self.tab2,textvariable=self.yOff) # ROI input
        self.yoffset.place(x=20, y=195, width=50)
        
        if not torch.cuda.is_available():
            self.start_live_decon['state'] = DISABLED
            print('A valid GPU is required for live ML-SIM')
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
        
    def start_live(self):
        
        optosplit = self.opto.get()
        if optosplit == 1:
            x1 = self.x1.get() # get ROI variables from the GUI input
            y1 = self.y1.get()
            x2 = self.x2.get() # get ROI variables from the GUI input
            y2 = self.y2.get()
            with Bridge() as bridge: # load camera control library
                xmin = min(x1,x2)
                xmax = max(x1,x2)
                width = xmax-xmin+512
                ymin = min(y1,y2)
                ymax = max(y1,y2)
                height = ymax-ymin+512
                core = bridge.get_core()
                ROI = [xmin, ymin, width, height] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI')
        else:
            with Bridge() as bridge: # load camera control library
                x1 = self.xOff.get() # get ROI variables from the GUI input
                y1 = self.yOff.get()
                x2 = 0
                y2 = 0
                core = bridge.get_core()
                ROI = [x1, y1, 512, 512] # build ROI 
                core.set_roi(*ROI) # set ROI  
                print('Successfully set ROI') 


        exposure_time = self.expTime.get()
        opto = self.opto.get()
        self.live_process = mp.Process(target= asf.live_view, args = (self.stop_signal,self.output,exposure_time,opto,x1,y1,x2,y2))
        self.live_process.start()
        self.plotting_process = threading.Thread(target= self.plot)
        self.plotting_process.start()

    def start_ml_sim(self):
        exposure_time = self.expTime.get()
        self.live_process = mp.Process(target= asf.live_ml_sim, args = (self.stack,self.stop_signal,self.output,exposure_time))
        self.live_process.start()
        self.plotting_process = threading.Thread(target= self.plot)
        self.plotting_process.start()    

    def quit_gui(self):
        self.stop_signal.put(False)
        time.sleep(1)
        self.master.destroy()

    def stop_live(self):
        self.stop_signal.put(False)

    def plot(self):
        while True: 
            if not self.output.empty():
                image_array = self.output.get() # empty data from reconstruction pool
                if isinstance(image_array, bool):
                    print('finished acquisition')
                    break
                else:
                    # run the update function
                    image_array = image_array-np.amin(image_array)
                    image_array = image_array*(255/np.amax(image_array)) 
                    image_array = image_array.astype('uint8')
                    img =  ImageTk.PhotoImage(image=Image.fromarray(image_array,mode='L')) # convert numpy array to tikner object 
                    self.panel.configure(image=img) # update the GUI element
                    self.panel.image = img  
            # else:
                # print('imArray was empty')

if __name__ == '__main__':
    root = tk.Tk()
    my_gui = ML_App(root)
    root.mainloop()

