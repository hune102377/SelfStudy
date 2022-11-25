import tkinter as tk
import tkinter.ttk 
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import messagebox

import mosaic

# import practice

main_window = tk.Tk()
main_window.title('GoV')
main_window.minsize(400, 300)  # 최소 사이즈

val_file_path = ''
val_save_path = ''
val_yolo_type = 'yolov5n'
val_taget_size = 480
val_conf_thres = 3
val_iou_thres = 5
val_crf_thres = 25



'''기능 추가'''
# 기능1 : 파일 선택
def select_file():
    global file_path
    try:
        filename = askopenfilename(initialdir="./", filetypes=(("Video files", ".avi .mp4"), ('All files', '*.*')))
        if filename:
            listbox1.delete(0, "end")
            listbox1.insert(0, filename)
            file_path = filename
    except:
        messagebox.showerror("Error", "오류가 발생했습니다.")

# 기능2 : 폴더 선택
def select_directory():
    global save_path
    try:
        foldername = askdirectory(initialdir="./")
        if foldername:
            listbox2.delete(0, "end")
            listbox2.insert(0, foldername)
            save_path = foldername
    except:
        messagebox.showerror("Error", "오류가 발생했습니다.")

# 기능3 : 변환 버튼
def select_convert(file_path, save_path):

    global val_yolo_type
    global val_taget_size
    global val_taget_size
    global val_conf_thres
    global val_iou_thres

    try:
        mosaic.mosaic(file_path, save_path, yolo_type = val_yolo_type, target_size = val_taget_size,
        conf_thres = val_conf_thres / 10, iou_thres = val_iou_thres / 10)
    except:
        messagebox.showerror("Error", "오류가 발생했습니다.")

# 기능4 : 실행 버튼
def select_practice(file_path):
    try:
        print("ui.py select_practice")
        # url=file_path
        # practice.practice(file_path)
    except:
        messagebox.showerror("Error", "오류가 발생했습니다.")


# 기능5: 세팅 버튼
def select_setting():
    global val_yolo_type
    global val_iou_thres
    global val_conf_thres
    global val_taget_size
    global val_crf_thres

    setting = tk.Toplevel()
    setting.geometry("275x450")

    def radio_yolo_type() :
        global val_yolo_type
        if box_yolov5_type.get() == 'yolov5n' :
            print(box_yolov5_type.get())
            val_yolo_type = 'yolov5n'
        elif box_yolov5_type.get() == 'yolov5m' :
            val_yolo_type = 'yolov5m'
            print(box_yolov5_type.get())    
        elif box_yolov5_type.get() == 'yolov5l' :
            val_yolo_type = 'yolov5l'
            print(box_yolov5_type.get())
    
    def radio_target_type() :
        global val_yolo_type
        if box_target_size.get() == 480 :
            val_yolo_type = 480
        elif box_target_size.get() == 720 :
            val_yolo_type = 720
        elif box_target_size.get() == 1080 :
            val_yolo_type = 1080

    def silder_iou() :
        global val_iou_thres
        val_iou_thres = button_var_iou.get()
        
    def silder_conf() :
        global val_conf_thres
        val_conf_thres = button_var_iou.get()
    
    def silder_crf() :
        global val_crf_thres
        val_crf_thres = button_var_iou.get()


    # 프레임 생성
    frm1 = tk.LabelFrame(setting, text="yolov5_type", pady=10, padx=15)   # pad 내부
    frm1.grid(row=0, column=1, pady=10, padx=10, sticky="nswe") # pad 내부

    frm2 = tk.LabelFrame(setting, text="target_size", pady=10, padx=15)   # pad 내부
    frm2.grid(row=1, column=1, pady=10, padx=10, sticky="nswe") # pad 내부

    frm3 = tk.LabelFrame(setting, text="var_iou", pady=5, padx=15)   # pad 내부
    frm3.grid(row=2, column=1, pady=5, padx=10, sticky="nswe") # pad 내부

    frm4 = tk.LabelFrame(setting, text="var_conf", pady=5, padx=15)   # pad 내부
    frm4.grid(row=3, column=1, pady=5, padx=10, sticky="nswe") # pad 내부

    frm5 = tk.LabelFrame(setting, text="var_crf", pady=5, padx=15)   # pad 내부
    frm5.grid(row=4, column=1, pady=5, padx=10, sticky="nswe") # pad 내부

    # 버튼 생성
    box_yolov5_type = tk.StringVar()
    button_yolov5_type_n = tk.Radiobutton(frm1, text="yolov5n", value='yolov5n', variable=box_yolov5_type, command = lambda : radio_yolo_type())
    button_yolov5_type_m = tk.Radiobutton(frm1, text="yolov5m", value='yolov5m', variable=box_yolov5_type, command = lambda : radio_yolo_type())
    button_yolov5_type_l = tk.Radiobutton(frm1, text="yolov5l", value='yolov5l', variable=box_yolov5_type, command = lambda : radio_yolo_type())
    
    if val_yolo_type == 'yolov5n' :
        button_yolov5_type_n.select()
    elif val_yolo_type == 'yolov5m' :
        button_yolov5_type_m.select()
    elif val_yolo_type == 'yolov5l' :
        button_yolov5_type_l.select()

    box_target_size = tk.IntVar()
    button_var_target_size_480 = tk.Radiobutton(frm2, text="480", value=480, variable=box_target_size, command = lambda : radio_target_type())
    button_var_target_size_720 = tk.Radiobutton(frm2, text="720", value=720, variable=box_target_size, command = lambda : radio_target_type())
    button_var_target_size_1080 = tk.Radiobutton(frm2, text="1080", value=1080, variable=box_target_size, command = lambda : radio_target_type())

    if val_taget_size == 480 :
        button_var_target_size_480.select()
    elif val_taget_size == 720 :
        button_var_target_size_720.select()
    elif val_taget_size == 1080 :
        button_var_target_size_1080.select()

    button_var_iou = tk.Scale(frm3, from_=0, to=10, orient="horizontal", length=200, command = lambda x : silder_iou())
    button_var_iou.set(val_iou_thres)
    
    button_var_conf = tk.Scale(frm4, from_=0, to=10, orient="horizontal", length=200, command = lambda x : silder_conf())
    button_var_conf.set(val_conf_thres)

    button_var_crf = tk.Scale(frm5, from_=0, to=15, orient="horizontal", length=200, command = lambda x : silder_crf())
    button_var_crf.set(25 - val_crf_thres)


    # 버튼 형성
    button_yolov5_type_n.grid(row=0, column=0)
    button_yolov5_type_m.grid(row=0, column=1)
    button_yolov5_type_l.grid(row=0, column=2)
    
    button_var_target_size_480.grid(row=1, column=0)
    button_var_target_size_720.grid(row=1, column=1)
    button_var_target_size_1080.grid(row=1, column=2)

    button_var_iou.grid(row=2, column=1)
    button_var_conf.grid(row=3, column=1)
    button_var_crf.grid(row=4, column=1)

    
def select_radio() :
    global val_yolo_type
    val_yolo_type = 'yolov5n'

'''1. 프레임 생성'''
# 상단 프레임 (LabelFrame)
frm1 = tk.LabelFrame(main_window, text="준비", pady=15, padx=15)   # pad 내부
frm1.grid(row=0, column=0, pady=10, padx=10, sticky="nswe") # pad 내부
main_window.columnconfigure(0, weight=1)   # 프레임 (0,0)은 크기에 맞춰 늘어나도록
main_window.rowconfigure(0, weight=1)

# 하단 프레임 (Frame)
frm2 = tk.Frame(main_window, pady=10)
frm2.grid(row=1, column=0, pady=10)

'''2. 요소 생성'''
# 레이블
lbl1 = tk.Label(frm1, text='Select File')
lbl2 = tk.Label(frm1, text='Select Path')

# 리스트박스
listbox1 = tk.Listbox(frm1, width=40, height=1)
listbox2 = tk.Listbox(frm1, width=40, height=1)

# 버튼
btn1 = tk.Button(frm1, text="찾아보기", width=8, command=select_file)
btn2 = tk.Button(frm1, text="찾아보기", width=8, command=select_directory) 
btn3 = tk.Button(frm1, text="변환", width=8, height=4, command=lambda : select_convert(file_path, save_path)) 
btn4 = tk.Button(frm1, text="세팅", width=4, height=2, command=select_setting) 
btn5 = tk.Button(frm1, text="practice", width=8, height=4, command=lambda : select_practice(file_path)) 


'''3. 요소 배치'''
# 상단 프레임
lbl1.grid(row=0, column=0, sticky="e")
lbl2.grid(row=1, column=0, sticky="e", pady= 20)
listbox1.grid(row=0, column=1, columnspan=2, sticky="we")
listbox2.grid(row=1, column=1, columnspan=2, sticky="we")
btn1.grid(row=0, column=3)
btn2.grid(row=1, column=3)
btn3.grid(row=2, column=1)
btn4.grid(row=3, column=3)
btn5.grid(row=2, column=3)
# 상단프레임 grid (2,1)은 창 크기에 맞춰 늘어나도록
frm1.rowconfigure(2, weight=1)      
frm1.columnconfigure(1, weight=1)   

'''실행'''
main_window.mainloop()