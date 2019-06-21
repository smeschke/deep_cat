from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import cv2, os, numpy as np
import time

# Start time
start_time = time.time()

# Make Model: https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb
class Config(Config):
	NAME = "deep_segment"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 81
config = Config()
model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True) # https://github.com/matterport/Mask_RCNN/releases

img_path = '/home/stephen/Desktop/images/cat10.jpg'
img = cv2.imread(img_path)
src = img.copy()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w, _ = img.shape
dl_size = 512
dl_scale = w/dl_size

# Model has been created
model_creation_time = time.time()
print("Time for model creation: ", model_creation_time - start_time)

# Function to get mask from MRCNN
def get_deep_mask(img):
        img_dl_size = cv2.resize(img, (dl_size, int(h/dl_scale)))
        # Use DL to get a mask
        deep_mask = np.zeros_like(img_dl_size)
        results = model.detect([img_dl_size], verbose=1)
        r = results[0]
        mask = r["masks"][:, :, 0]        
        for i in range(0, len(r["scores"])):
                # Get the mask for this object
                mask = r["masks"][:,:,i]
                # Get the bounding box for this object
                roi = r["rois"][i]
                # If the object has a cat
                if r["class_ids"][i] == 16:
                #if r["class_ids"][i] > 0 and r["class_ids"][i] < 32:
                        #Line 72 - https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
                        deep_mask = visualize.apply_mask(deep_mask, mask, (255,255,255), alpha=.1)
                        # Stop after only one cat
                        break
        # Make things black and white
        _, deep_mask = cv2.threshold(deep_mask, 12, 255, cv2.THRESH_BINARY)
        return deep_mask, roi

# Get the mask and roi from the image
deep_mask, (y1,x1,y2,x2) = get_deep_mask(img_rgb)

# First pass has been completed
first_pass_complete_time = time.time()
print("Time for first pass: ", first_pass_complete_time - model_creation_time)

# Scale ROI values to the source image size
y1,x1,y2,x2 = np.array((y1*dl_scale,x1*dl_scale,y2*dl_scale,x2*dl_scale), int)
print("ROI VALUES: ", y1,x1,y2,x2)
# Add a buffer of 50 px
buffer = int(min(abs(x1-x2), abs(y1-y2))/5)
if y1-buffer >= 0: y1-= buffer
else: y1 = 0
if y2+buffer <= h: y2+= buffer
else: y2 = h
if x1-buffer >= 0: x1-=buffer
else: x1 = 0
if x2+buffer <= w: x2+=buffer
else: x2 = w
# Crop out the relevant part of the image
img = img[y1:y2, x1:x2]
img_rgb = img_rgb[y1:y2, x1:x2]
h,w,_ = img.shape
# Get the mask again, this time using only the roi
deep_mask, (y1,x1,y2,x2) = get_deep_mask(img_rgb)

# Second pass has been completed
second_pass_complete_time = time.time()
print("Time for Second pass: ", second_pass_complete_time - first_pass_complete_time)

print("APPLYING GRABCUT...............")
deep_mask = cv2.cvtColor(deep_mask, cv2.COLOR_BGR2GRAY)
deep_mask = cv2.resize(deep_mask, (w,h))
mask = np.zeros(img.shape[:2],np.uint8)
white_background = (255 - mask.copy())

# Initialize parameters for the GrabCut algorithm
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
iters, size = 4, int(h*w/124321)
print("Using a kernel size of: ", size)
kernel = np.ones((size,size),np.uint8)
big_kernel = np.ones((2*size,2*size),np.uint8)
huge_kernel = np.ones((4*size,4*size),np.uint8)
# Dilate the mask to make sure the whole object is covered by the mask
dilation = cv2.dilate(deep_mask, big_kernel, iterations = iters)
# Start with a white background and subtract 
sure_background = white_background - dilation

# Erode to find the sure foreground
sure_foreground = cv2.erode(deep_mask, kernel, iterations = iters)

# Change the values on the mask so that:
#    2 - unsure pixels
#    1 - sure foreground pixels
#    0 - sure background pixels
mask[:] = 2
mask[sure_background == 255] = 0
mask[sure_foreground == 255] = 1

# Apply GrabCut
out_mask = mask.copy()
out_mask, _, _ = cv2.grabCut(img,out_mask,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
out_mask = np.where((out_mask==2)|(out_mask==0),0,1).astype('uint8')
# Open the mask to fill in the holes
out_img = img*out_mask[:,:,np.newaxis]

# First pass has been completed
grabcut_complete_time = time.time()
print("Time for GrabCut: ", grabcut_complete_time - second_pass_complete_time)

# Plot with Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create image that shows foreground and background
def create_labeled_image(src_image, foreground, background):
        for i in range(4):
                bg = np.zeros_like(img)
                bg[background == 0] = (255,255,255)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                _,thresh = cv2.threshold(bg,1,255,cv2.THRESH_BINARY)
                contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                src_image=cv2.drawContours(src_image, contours, -1, (0,255,255), 5-i)
                background = cv2.erode(background, huge_kernel, iterations = 1)
        for i in range(4):
                bg = np.zeros_like(img)
                bg[:,:,:] = 255,255,255
                bg[foreground == 0] = (0,0,0)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                _,thresh = cv2.threshold(bg,1,255,cv2.THRESH_BINARY)
                contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                src_image=cv2.drawContours(src_image, contours, -1, (255,255,0), 5-i)
                foreground = cv2.erode(foreground, huge_kernel, iterations = 1)
        return src_image


# Create a multi plot
f, axarr = plt.subplots(2,3, sharex=True)
# Show source image in the top left
src_h, src_w, _ = src.shape
scale_w = int(w/3)
scale_h = int(scale_w * (src_h/src_w))
src = cv2.resize(src, (scale_w, scale_h))
comp_image = img.copy()
comp_image[:scale_h,:scale_w] = src
axarr[0,0].imshow(comp_image)
# Show deep mask in the top middle
axarr[0,1].imshow(deep_mask)
# Show deep mask of source image in the top right
deep_mask_of_source = img.copy()
deep_mask_of_source[deep_mask == 0] = 0,0,0
axarr[0,2].imshow(deep_mask_of_source)
# Show the sure foreground and the sure background in the bottom left
labeled_image = create_labeled_image(img, sure_foreground, sure_background)
axarr[1,0].imshow(labeled_image)
# Show the GrabCut mask in the bottom middle
axarr[1,1].imshow(out_mask)
# Show the GrabCut image in the bottom right
axarr[1,2].imshow(out_img)

# Add titles
text = 'Source Image: '+str(src_w)+'x'+str(src_h)+' px'+' and ROI: '+str(w)+'x'+str(h)+' px'
axarr[0,0].set_title(text)
axarr[0,1].set_title('Mask from DL')
axarr[0,2].set_title('DL Mask Image: 512x512 px')
axarr[1,0].set_title('Sure Foreground and Sure Background')
axarr[1,1].set_title('GrabCut Mask')
axarr[1,2].set_title('GrabCut Mask Image')
# Clean up and show
axarr[0,0].axis('off')
axarr[0,1].axis('off')
axarr[1,0].axis('off')
axarr[1,1].axis('off')
axarr[1,2].axis('off')
axarr[0,2].axis('off')
plt.show()
