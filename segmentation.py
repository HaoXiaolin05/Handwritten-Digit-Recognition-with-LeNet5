import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from original import LeNet5 as LeNet5org
from improve import LeNet5 as LeNet5imp

IMAGE_PATH = "images/test6.jpg"
num = 235

def predit_line_of_digits(image_path, device, modelOrg, transformOrg, modelImp, transformImp):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    img = cv2.medianBlur(img, 15)
    # img = cv2.GaussianBlur(img, (35, 35), 0)
    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap="gray")
    plt.title("Grayscale input Image")

    # just take the pixel has value from 128 to 255
    _, thresh_org = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # use delating to make the digit bigger -> use for contouring
    thresh_org = cv2.morphologyEx(thresh_org, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))
    plt.subplot(232)
    plt.imshow(thresh_org, cmap="gray")
    plt.title("Morphological dilation (7x7)")
    # use closing to connect components (small gap)
    thresh = cv2.morphologyEx(thresh_org, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    plt.subplot(233)
    plt.imshow(thresh, cmap="gray")
    plt.title("Morphological closing (9x9)")

    # find contours (each segmentation) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
    plt.subplot(234)
    plt.imshow(img, cmap="gray")
    plt.title("Boundaries of digits (after contour analysis)")

    # sort contours from left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes.sort(key=lambda x: x[0])
    print(bounding_boxes)
    full_number_string_org = ""
    full_number_string_imp = ""
    for (x, y, w, h) in bounding_boxes:
        # crop into the digit
        if w * h < 5000: continue # remove smaller box (could be the noise)
        pad = 5
        roi = thresh_org[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

        for r in split_digit(roi):
            # convert it to pil to perform the resize and predict
            roi_pil = Image.fromarray(r)
            digit_input = resize_and_pad(roi_pil)
            
            tensor_input_org = transformOrg(digit_input).unsqueeze(0).to(device)
            tensor_input_imp = transformImp(digit_input).unsqueeze(0).to(device)
            with torch.no_grad():
                output_org = modelOrg(tensor_input_org)
                prediction_org = output_org.argmax(dim=1).item()
                full_number_string_org += str(prediction_org)
                output_imp = modelImp(tensor_input_imp)
                prediction_imp = output_imp.argmax(dim=1).item()
                full_number_string_imp += str(prediction_imp)

    return full_number_string_org, full_number_string_imp

def split_digit(img_roi):
    global num
    h, w = img_roi.shape[:2]
    
    if w / float(h) > 1.1:
        # calculate the the distance transform (the logic from the watershed algorithm)
        distTrans = cv2.distanceTransform(img_roi, cv2.DIST_L2, 5)
        
        _, distThresh = cv2.threshold(distTrans, 15, 255, cv2.THRESH_BINARY)
        distThresh_8u = np.uint8(distThresh)
        
        contours, _ = cv2.findContours(distThresh_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [img_roi]

        # get the largest contour to split
        c = max(contours, key=cv2.contourArea)
        x_core, y_core, w_core, h_core = cv2.boundingRect(c)
        
        # center line (at the middle of the contour)
        split_x = x_core + (w_core // 2)
        
        overlap = 15
        
        # left cut
        cut_left_end = min(w, split_x + overlap)
        part1 = img_roi[:, 0:cut_left_end]
        
        # right cut
        cut_right_start = max(0, split_x - overlap)
        part2 = img_roi[:, cut_right_start:w]

        plt.subplot(num)
        num = num + 1
        plt.imshow(img_roi, cmap='gray')
        plt.axvline(x=split_x, color='r', linestyle='--')
        plt.axvline(x=cut_left_end, color='g', linestyle=':') # Green dotted = Left img end
        plt.axvline(x=cut_right_start, color='b', linestyle=':') # Blue dotted = Right img start
        plt.title("The cut line")
        
        if part1.shape[1] > 0 and part2.shape[1] > 0:
            return [part1, part2]

    return [img_roi]

def resize_and_pad(img_pil):
    w, h = img_pil.size

    # convert it around 20 x 20
    ratio = min(20/w, 20/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

    # create the background
    final_img = Image.new("L", (28, 28), 0)
    
    # put the digit on the background
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    final_img.paste(img_resized, (paste_x, paste_y))
    
    return final_img

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelOrg = load_model(LeNet5org, "lenet_original.pth")
    modelImp = load_model(LeNet5imp, "lenet_improve.pth")

    transformOrg = transforms.Compose([transforms.ToTensor()])
    transformImp = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    org, imp = predit_line_of_digits(IMAGE_PATH, device, modelOrg, transformOrg, modelImp, transformImp)
    print(f"Original Model: {org}")
    print(f"Improved Model: {imp}")
    plt.show()