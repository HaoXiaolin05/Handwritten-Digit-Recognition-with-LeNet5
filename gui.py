import tkinter as tk
from PIL import Image, ImageDraw, ImageTk 
import torch
import torchvision.transforms as transforms
from original import LeNet5 as LeNet5org
from improve import LeNet5 as LeNet5imp

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_x, last_y = None, None

def load_model(model_class, path):
    try:
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Loaded {path}")
        return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

modelOrg = load_model(LeNet5org, "lenet_original.pth")
modelImp = load_model(LeNet5imp, "lenet_improve.pth")

# Image that on canvas (feed to model)
image1 = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), color=0)
draw = ImageDraw.Draw(image1)

def activate_paint(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def paint(event):
    global last_x, last_y
    x, y = event.x, event.y

    brush_width = size_slider.get()

    if last_x and last_y:
        # Draw on GUI Canvas
        canvas.create_line((last_x, last_y, x, y), 
                           width=brush_width, 
                           fill="white", 
                           capstyle=tk.ROUND, 
                           smooth=True, 
                           splinesteps=36)
        
        # Draw on the image feed two to models
        draw.line([last_x, last_y, x, y], fill=255, width=brush_width, joint="curve")

    last_x, last_y = x, y

def process_and_predict():
    global preview_photo_ref

    img_28x28 = image1.resize((28, 28), resample=Image.BILINEAR) # Resize it to 28x28 (size of MNIST dataset)

    img_preview = img_28x28.resize((150, 150), resample=Image.NEAREST)
    preview_photo_ref = ImageTk.PhotoImage(img_preview)
    lbl_preview_img.config(image=preview_photo_ref)
    lbl_preview_text.config(text="Status: Resized to 28x28")

    # Transform like the two in the .py model files
    transformOrg = transforms.Compose([transforms.ToTensor()])
    transformImp = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Add Batch dim .unsqueeze(0)
    # Shape becomes: [1, 1, 28, 28]
    if modelOrg:
        input_org = transformOrg(img_28x28).unsqueeze(0).to(device)
        predict_single(modelOrg, input_org, lbl_resultOrg, lbl_confOrg)
    
    if modelImp:
        input_imp = transformImp(img_28x28).unsqueeze(0).to(device)
        predict_single(modelImp, input_imp, lbl_resultImp, lbl_confImp)

def predict_single(model, tensor_input, label_res, label_conf):
    try:
        with torch.no_grad():
            output = model(tensor_input)
            # Softmax uses to get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            
            prediction = probs.argmax(dim=1).item()
            confidence = probs[0][prediction].item()

            label_res.config(text=f"Prediction: {prediction}", fg="blue")
            label_conf.config(text=f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(e)
        label_res.config(text="Error")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_WIDTH, CANVAS_HEIGHT], fill=0)
    lbl_resultOrg.config(text="Prediction: ?")
    lbl_confOrg.config(text="")
    lbl_resultImp.config(text="Prediction: ?")
    lbl_confImp.config(text="")

root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("900x700")

# Left Frame
frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, padx=20, pady=20)

# The Canvas (Now 280x280)
canvas = tk.Canvas(frame_left, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black", cursor="cross")
canvas.pack(pady=10)

# Tools Section
tools_frame = tk.Frame(frame_left)
tools_frame.pack(pady=10, fill=tk.X)

# Tools
size_slider = tk.Scale(tools_frame, from_=10, to=50, orient=tk.HORIZONTAL, length=250)
size_slider.set(30) # default
size_slider.pack(side=tk.LEFT, padx=10)

btn_clear = tk.Button(tools_frame, text="Clear All", width=8, command=clear_canvas, bg="#ffcccb")
btn_clear.pack(side=tk.RIGHT, padx=5)


# Right Frame
frame_right = tk.Frame(root, relief=tk.RIDGE, borderwidth=2)
frame_right.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

tk.Label(frame_right, text="Analysis", font=("Arial", 14, "bold")).pack(pady=10)
tk.Button(frame_right, text="PREDICT", font=("Arial", 12, "bold"), bg="lightgreen", command=process_and_predict).pack(pady=10)

# Preview
tk.Label(frame_right, text="Input (Resized to 28x28):", font=("Arial", 10)).pack()
lbl_preview_img = tk.Label(frame_right, bg="gray")
lbl_preview_img.pack(pady=5)
lbl_preview_text = tk.Label(frame_right, text="Status: Waiting...", fg="gray")
lbl_preview_text.pack()

# Results
tk.Label(frame_right, text="--------- Original Model ---------", fg="gray").pack(pady=(20, 5))
lbl_resultOrg = tk.Label(frame_right, text="Prediction: ?", font=("Arial", 20, "bold"))
lbl_resultOrg.pack()
lbl_confOrg = tk.Label(frame_right, text="", font=("Arial", 11))
lbl_confOrg.pack()

tk.Label(frame_right, text="--------- Improved Model ---------", fg="gray").pack(pady=(20, 5))
lbl_resultImp = tk.Label(frame_right, text="Prediction: ?", font=("Arial", 20, "bold"))
lbl_resultImp.pack()
lbl_confImp = tk.Label(frame_right, text="", font=("Arial", 11))
lbl_confImp.pack()

# Event action
canvas.bind("<Button-1>", activate_paint)
canvas.bind("<B1-Motion>", paint)

root.mainloop()