import tkinter
import tkinter.messagebox
import customtkinter
from screeninfo import get_monitors
from customtkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from Product_Photography import *
from PIL import Image, ImageDraw, ImageFont
from tkinter import messagebox 


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

bg_change_generator = BackgroundChangeGenerator()

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        monitors = get_monitors()
        primary_monitor = monitors[0]
        width, height = primary_monitor.width, primary_monitor.height

        app_width = width
        app_height = height

        # configure window
        self.title("Product Photography with generative A.I")
        self.geometry(f"{app_width}x{app_height}+0+0")

        # configure grid layout (4x4)

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3,4,5), weight=1)
        self.grid_rowconfigure((0, 1, 2 ,3,4,5), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Product Photography", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text="Load Image", command=self.load_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,text="Start", command=self.save_annotation)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Clear Boxes" ,command=self.clear_boxes)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Refresh" ,command=self.refresh_all)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Save Output" ,command=self.save_image_to_downloads)
        self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)

        self.prompt = customtkinter.CTkEntry(self, placeholder_text="Prompt to change background")
        self.prompt.grid(row=3, column=1, columnspan=1, padx=(20, 0), pady=(20, 20), sticky="nsew")
        
        self.submit = customtkinter.CTkButton(master=self, fg_color="transparent", text="Generate" , border_width=2,command=lambda: self.print_prompt(),text_color=("gray10", "#DCE4EE"))
        self.submit.grid(row=3, column=2, padx=(0, 10), pady=(20, 20))
        
        self.input_frame_canvas = customtkinter.CTkCanvas(self, width=600,height = 600, scrollregion=(0, 0, 2000, 5000))
        self.input_frame_canvas.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.input_frame_canvas_scrollbar_y = customtkinter.CTkScrollbar(self.input_frame_canvas, command=self.input_frame_canvas.yview)
        self.input_frame_canvas_scrollbar_y.pack(side="right", fill="y")

        self.input_frame_canvas.configure(yscrollcommand=self.input_frame_canvas_scrollbar_y.set)

        self.input_frame_canvas_scrollbar_x = customtkinter.CTkScrollbar(self.input_frame_canvas, width=600 , command=self.input_frame_canvas.xview, orientation="horizontal")
        self.input_frame_canvas_scrollbar_x.pack(side="bottom", fill="x")

        self.input_frame_canvas.configure(xscrollcommand=self.input_frame_canvas_scrollbar_x.set)
        
        self.input_frame_label = customtkinter.CTkLabel(self.input_frame_canvas, text="Input Raw Frame", fg_color=("gray75", "white"),width=630)
        self.input_frame_label.place(x=0, y=0)
        
        self.input_frame_canvas.bind("<ButtonPress-1>", self.start_box)
        self.input_frame_canvas.bind("<B1-Motion>", self.draw_box)
        self.input_frame_canvas.bind("<ButtonRelease-1>", self.end_box)
        
        
        self.binary_frame_canvas = customtkinter.CTkCanvas(self, width=600,height = 600, scrollregion=(0, 0, 2000, 5000))
        self.binary_frame_canvas.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.binary_frame_canvas_scrollbar_y = customtkinter.CTkScrollbar(self.binary_frame_canvas, command=self.binary_frame_canvas.yview)
        self.binary_frame_canvas_scrollbar_y.pack(side="right", fill="y")

        self.binary_frame_canvas.configure(yscrollcommand=self.binary_frame_canvas_scrollbar_y.set)

        self.binary_frame_canvas_scrollbar_x = customtkinter.CTkScrollbar(self.binary_frame_canvas, width=600 , command=self.binary_frame_canvas.xview, orientation="horizontal")
        self.binary_frame_canvas_scrollbar_x.pack(side="bottom", fill="x")

        self.binary_frame_canvas.configure(xscrollcommand=self.binary_frame_canvas_scrollbar_x.set)
        
        self.binary_fram_label = customtkinter.CTkLabel(self.binary_frame_canvas, text="Binary Output Results", fg_color=("gray75", "white"),width=630)
        self.binary_fram_label.place(x=0, y=0)
        
        
        self.background_rm_frame_canvas = customtkinter.CTkCanvas(self, width=600,height = 600, scrollregion=(0, 0, 2000, 5000))
        self.background_rm_frame_canvas.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.background_rm_frame_canvas_scrollbar_y = customtkinter.CTkScrollbar(self.background_rm_frame_canvas, command=self.background_rm_frame_canvas.yview)
        self.background_rm_frame_canvas_scrollbar_y.pack(side="right", fill="y")

        self.background_rm_frame_canvas.configure(yscrollcommand=self.background_rm_frame_canvas_scrollbar_y.set)

        self.background_rm_frame_canvas_scrollbar_x = customtkinter.CTkScrollbar(self.background_rm_frame_canvas, width=600 , command=self.background_rm_frame_canvas.xview, orientation="horizontal")
        self.background_rm_frame_canvas_scrollbar_x.pack(side="bottom", fill="x")

        self.background_rm_frame_canvas.configure(xscrollcommand=self.background_rm_frame_canvas_scrollbar_x.set)
        
        self.background_rm_fram_label = customtkinter.CTkLabel(self.background_rm_frame_canvas, text="Background Removed Results", fg_color=("gray75", "white"),width=630)
        self.background_rm_fram_label.place(x=0, y=0)
        
        
        self.output_frame_canvas = customtkinter.CTkCanvas(self, width=600,height = 600, scrollregion=(0, 0, 2000, 5000))
        self.output_frame_canvas.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        self.output_frame_canvas_scrollbar_y = customtkinter.CTkScrollbar(self.output_frame_canvas, command=self.output_frame_canvas.yview)
        self.output_frame_canvas_scrollbar_y.pack(side="right", fill="y")

        self.output_frame_canvas.configure(yscrollcommand=self.output_frame_canvas_scrollbar_y.set)

        self.output_frame_canvas_scrollbar_x = customtkinter.CTkScrollbar(self.output_frame_canvas, width=600 , command=self.output_frame_canvas.xview, orientation="horizontal")
        self.output_frame_canvas_scrollbar_x.pack(side="bottom", fill="x")

        self.output_frame_canvas.configure(xscrollcommand=self.output_frame_canvas_scrollbar_x.set)
        
        self.output_fram_label = customtkinter.CTkLabel(self.output_frame_canvas, text="Output Results", fg_color=("gray75", "white"),width=630)
        self.output_fram_label.place(x=0, y=0)
        
        self.image = Image.open('./hello_world.png')
        self.photo = ImageTk.PhotoImage(self.image)
        self.input_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.input_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        
        self.output_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.output_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))

        self.background_rm_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.background_rm_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        
        self.binary_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.binary_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
    
        self.current_box = None
        self.start_x, self.start_y = None, None
        self.rectangles = []
        self.created_boxes = []

        self.image = None
        self.image_path = None
        self.mask = None
        self.input_image = None
        self.stable_diffusion_output = None
        
    def print_prompt(self):
        if self.mask is None or self.input_image is None:
            messagebox.showwarning("Warning", "Please start the process by removing background") 
        else:
            print('prompt done')
            self.stable_diffusion_output = bg_change_generator.generate_background_change_api(
                prompt=self.prompt.get().strip(),
                image=self.input_image,
                mask=self.mask,
                output_path="./out.png",
                key='sk-y1SMimJTiCmiW7RvHnSYxZGb8zbo2jr4xdmW5M8eOs9y172D',
                engine="stable-diffusion-xl-1024-v1-0"
            )
            self.stable_diffusion_output_image = ImageTk.PhotoImage(self.stable_diffusion_output)
            self.output_frame_canvas.create_image(0, 0, image=self.stable_diffusion_output_image, anchor=tk.NW)
            self.output_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
            self.prompt.configure(placeholder_text="Prompt to change background")
            prompt_data = self.prompt.get().strip()
            messagebox.showinfo(
                title = "Success",
                message = f"Image generated with prompt : {prompt_data}",
            )
        
    def save_image_to_downloads(self):
        if self.stable_diffusion_output is None:
            messagebox.showwarning("Warning", "Draw the area to be removed from background") 
        else:    
            downloads_path = bg_change_generator.get_downloads_path()
            self.stable_diffusion_output.save(os.path.join(downloads_path, 'stable_diffusion_output.png'))
            messagebox.showinfo(
                title = "Success",
                message = f"Image succesfully saved in {downloads_path}",
            )
                        
    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")
        
    def combobox_callback(self,choice):
        print("combobox dropdown clicked:", choice)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif *.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            self.image_path = file_path
            self.photo = ImageTk.PhotoImage(self.image)
            self.input_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.input_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
            
    def start_box(self, event):
        x = self.input_frame_canvas.canvasx(event.x)
        y = self.input_frame_canvas.canvasy(event.y)
        self.start_x, self.start_y = x, y
        self.current_box = self.input_frame_canvas.create_rectangle(x, y, x, y, outline="red", width=2)
    
    def draw_box(self, event):
        x = self.input_frame_canvas.canvasx(event.x)
        y = self.input_frame_canvas.canvasy(event.y)
        self.input_frame_canvas.coords(self.current_box, self.start_x, self.start_y, x, y)
    
    def end_box(self, event):
        x = self.input_frame_canvas.canvasx(event.x)
        y = self.input_frame_canvas.canvasy(event.y)
        print((self.start_x, self.start_y, x, y))
        self.rectangles.append([self.start_x, self.start_y, x, y])
        self.created_boxes.append(self.current_box)
        self.start_x, self.start_y = None, None
        self.current_box = None
        
    def clear_boxes(self):
        if messagebox.askretrycancel("Alert", "Error: Are you sure want to delete boxes?") == False:
            pass
        else:
            if self.rectangles:
                for create_box in self.created_boxes:
                    self.input_frame_canvas.delete(create_box)
                self.rectangles = []
                self.created_boxes = []
            else:
                print("No boxes to delete.")
    
    def refresh_all(self):
        if messagebox.askretrycancel("Alert", "Error: Are you sure want to refresh?") == False:
            pass
        else:
            self.input_frame_canvas.delete('all')
            self.output_frame_canvas.delete('all')
            self.annotated_image = None
            self.current_box = None
            self.start_x, self.start_y = None, None
            self.rectangles = []
            self.created_boxes = []
            self.image = None
            self.image_path = None
            self.mask_photo = None
            self.mask = None
            self.annotated_photo = None
            self.stable_diffusion_output = None
            
            self.image = Image.open('./hello_world.png')
            self.photo = ImageTk.PhotoImage(self.image)
            
            self.input_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.input_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
            
            self.output_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.output_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))

            self.background_rm_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.background_rm_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
            
            self.binary_frame_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.binary_frame_canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))
        
    def save_annotation(self):
        if self.image_path:
            if self.rectangles:
                for box in self.rectangles:
                    # draw.rectangle(box, outline="red", width=2)
                    # annotated_image.save("annotated_" + self.image_path.split("/")[-1])
                    everything_results = model(self.image_path, device='cuda', retina_masks=True, imgsz=640, conf=0.4, iou=0.9)
                    prompt_process = FastSAMPrompt(self.image_path, everything_results, device='cuda')
                    ann = prompt_process.box_prompt(bbox=box)
                    self.input_image,self.annotated_image,self.mask = bg_change_generator.get_image(ann)
                    draw = ImageDraw.Draw(self.annotated_image)
                    draw.rectangle(box, outline="red", width=2)
                # Update the instance variable with the new annotated photo
                self.annotated_photo = ImageTk.PhotoImage(self.annotated_image)
                
                self.mask_photo = ImageTk.PhotoImage(self.mask)
                # Update the annotated_canvas with the new image
                self.background_rm_frame_canvas.create_image(0, 0, image=self.annotated_photo, anchor=tk.NW)
                self.background_rm_frame_canvas.config(scrollregion=(0, 0, self.annotated_image.width, self.annotated_image.height))
                
                self.binary_frame_canvas.create_image(0, 0, image=self.mask_photo, anchor=tk.NW)
                self.binary_frame_canvas.config(scrollregion=(0, 0, self.mask.width, self.mask.height))
                
                messagebox.showinfo(
                    title = "Success",
                    message = "Process completed successfully.",
                )
                
                print("Annotations saved.")
            else:
                print("No image loaded.")   
                messagebox.showwarning("Warning", "Draw the area to be removed from background")      
        else:
            messagebox.showwarning("Warning", "Draw the area to be removed from background") 

if __name__ == "__main__":
    app = App()
    app.iconbitmap("logo.ico")
    model = FastSAM('FastSAM-s.pt') 
    app.mainloop()
    
