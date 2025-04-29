import utils
import ifm
import numpy as np
import cv2
from cam_hamamatsu4 import CTX_Hamamatsu, make_hamamatsu, destroy_hamamatsu
# print(np.version.version)
import time 

# Try initializing the camera with error handling
try:
    cam_ctx: CTX_Hamamatsu = make_hamamatsu(0, exposure_time=0.004)
except RuntimeError as e:
    print(f"❌ Camera initialization failed: {e}")
    exit(1)  # Exit if the camera fails to initialize

# Initialize IFM processing context
ifm_ctx: ifm.CTX = ifm.make(cam_ctx.width, cam_ctx.height)

# Image Processing Variables
h, w = cam_ctx.height, cam_ctx.width
h //= 2
w //= 2
y, x = np.indices((h, w))
y -= h // 2
x -= w // 2
k = 2 * np.pi / (780 * 1e-9)
L = 1e-2

a = ((np.sqrt(17) - 1) / 8)
a1 = np.arctan(1 / (1 - a))
a2 = 0.475

print("Angle a1:", a1)

# # OpenCV Window Setup
# cv2.namedWindow("Phase Overlay", cv2.WINDOW_NORMAL)

# # Ensure window size is valid before resizing
# if cam_ctx.width > 0 and cam_ctx.height > 0:
#     cv2.resizeWindow("Phase Overlay", cam_ctx.width, cam_ctx.height)

from PIL import Image
import numpy as np
# im1_atoms = Image.open(r'D:\LKB\off_axis_holography\time_lapse_2frames_no_delay_speed2_pulseBfieldOfieldoff400ms_on1600ms_50usdelaybwCamProbe28_00001.tif')
# im2_atoms = Image.open(r'D:\LKB\off_axis_holography\time_lapse_2frames_no_delay_speed2_pulseBfieldOfieldoff400ms_on1600ms_50usdelaybwCamProbe28_00002.tif')

# im1 = Image.open(r'D:\LKB\off_axis_holography\time_lapse_2frames_noMOT31_00001.tif')
# im2 = Image.open(r'D:\LKB\off_axis_holography\time_lapse_2frames_noMOT31_00002.tif')

# im1_atoms2 = Image.open(r'D:\LKB\off_axis_holography\no_delay_speed2_pulseBfieldOfieldoff400ms_on1600ms_50usdelaybwCamProbe32_00001.tif')
# im2_atoms2 = Image.open(r'D:\LKB\off_axis_holography\no_delay_speed2_pulseBfieldOfieldoff400ms_on1600ms_50usdelaybwCamProbe32_00002.tif')

# im.show()

# imarray1 = np.array(im1,  dtype=np.uint16)
# imarray2 = np.array(im2,  dtype=np.uint16)
# data = np.array((imarray1, imarray2))
# print('data size',np.size(data))
# p_stored = np.zeros(np.size(data))
def phase(img):
    try:
        for i in range(1):  # Continuous acquisition loop
            t3 = time.time()
            # frame = cam_ctx.acquire_frame()  # Capture frame from camera
            imarray1 = np.array(img,  dtype=np.uint16)
            frame = imarray1
            # frame2 = imarray2
            print('size',np.size(frame))
            # frame = np.zeros((500,500))
            t4 = time.time()
            print(frame.shape)
            if frame is None or frame.size == 0:
                print("⚠️ Warning: Empty frame received! Retrying...")
                continue  # Skip this frame

            # Compute Interferogram Field
            E = ifm.field_from_interferogram(ifm_ctx, frame) # get field once pbs is aligned, full fft as opp to overlay 
            p = np.angle(E)
            p[p < 0] += 2 * np.pi
            p = ifm.discretize(p, 8)
            print("max p", np.max(p))
        
            # Compute overlay
            # frame_discretized = ifm.discretize(frame, 8)
            t1 = time.time()
            overlay = ifm.overlay_cv2(ifm_ctx, frame) # real fft only
        
            print('time taken', t4-t3)
            # overlay = cv2.resize(overlay, (750, 500),0,0)
            # cv2.imshow("overlay", overlay)
            # t2 = time.time()
            # print('time', t2-t1, 'time first', t3-t1, 'time capture', t4-t1)
            # overlay_rescaled = utils.img_rescale_px(overlay, cam_ctx.width, cam_ctx.height)
            # amp = ifm.discretize(np.abs(E) ** 2, 8)

            
            # # Convert and display image
            # # amp_rescaled = np.array(amp, dtype=np.float32)
            # # amp_display = cv2.normalize(amp_rescaled, None, 0, 255, cv2.NORM_MINMAX)
            # # amp_display = amp_display.astype(np.uint8)
            
            ## Ensure p is a NumPy array
            p = np.array(p, dtype=np.float32)  # Convert to float32 before processing
            # Normalize phase values between 0 and 255
            p_display = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX)
            # Convert to uint8 for OpenCV compatibility
            p_display = np.clip(p_display, 0, 255).astype(np.uint8) # was clipping to red color only first 8 bits for R
            # Display the processed image
            p_display = cv2.resize(p_display, (500,500))
            p_colored = cv2.applyColorMap(p_display, cv2.COLORMAP_TWILIGHT_SHIFTED)
            # image = cv2.imshow("phase", p_colored)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup before exiting
        cv2.destroyAllWindows()
        destroy_hamamatsu(cam_ctx)
        print("✅ Cleanup complete. Camera safely released.")
    return p_colored



cv2.waitKey(0) 
cv2.destroyAllWindows() 

try:
    while True:  # Continuous acquisition loop
        t3 = time.time()
        frame = cam_ctx.acquire_frame()  # Capture frame from camera
        # imarray1 = np.array(im1,  dtype=np.uint16)
        # frame = imarray1
        # frame2 = imarray2
        # print('size',np.size(frame))
        # frame = np.zeros((500,500))
        t4 = time.time()
        # print(frame.shape)
        if frame is None or frame.size == 0:
            print("⚠️ Warning: Empty frame received! Retrying...")
            continue  # Skip this frame
        
        # # Compute Interferogram Field
        E = ifm.field_from_interferogram(ifm_ctx, frame) # get field once pbs is aligned, full fft as opp to overlay 
        p = np.angle(E)
        p[p < 0] += 2 * np.pi
        p = ifm.discretize(p, 8)
        print("max p", np.max(p))
       
        # Compute overlay
        # frame_discretized = ifm.discretize(frame, 8)
        # THIS:
        overlay = ifm.overlay_cv2(ifm_ctx, frame) # real fft only
       
        print('time taken', t4-t3)
        ## THIS:
        overlay = cv2.resize(overlay, (750, 500),0,0)
        cv2.imshow("overlay", overlay)
        
        t1 = time.time()
        print('tme', t4-t1, 'all', t3-t1)
        # t2 = time.time()
        # print('time', t2-t1, 'time first', t3-t1, 'time capture', t4-t1)
        # overlay_rescaled = utils.img_rescale_px(overlay, cam_ctx.width, cam_ctx.height)
        # amp = ifm.discretize(np.abs(E) ** 2, 8)

        
        # # Convert and display image
        # # amp_rescaled = np.array(amp, dtype=np.float32)
        # # amp_display = cv2.normalize(amp_rescaled, None, 0, 255, cv2.NORM_MINMAX)
        # # amp_display = amp_display.astype(np.uint8)
        
        ## Ensure p is a NumPy array
        p = np.array(p, dtype=np.float32)  # Convert to float32 before processing
        # Normalize phase values between 0 and 255
        p_display = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to uint8 for OpenCV compatibility
        p_display = np.clip(p_display, 0, 255).astype(np.uint8) # was clipping to red color only first 8 bits for R
        # Display the processed image
        p_display = cv2.resize(p_display, (500,500))
        p_colored = cv2.applyColorMap(p_display, cv2.COLORMAP_TWILIGHT_SHIFTED)
        # cv2.imshow("phase", p_colored)

        # ## Ensure amp is a NumPy array and convert to float32 before normalization
        # amp_rescaled = np.array(amp, dtype=np.float32)
        # # Normalize values between 0 and 255
        # amp_display = cv2.normalize(amp_rescaled, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert to uint8 (required for OpenCV)
        # amp_display = np.clip(amp_display, 0, 255).astype(np.uint8)
        # # Display the image
        # amp_display = cv2.resize(amp_display, (500,500))
        # # p = cv2.resize(p, (500,500))
        # # cv2.imshow("field amp", amp_display)

        # # # cv2.imshow("field amp", amp_display)
        # # cv2.imshow("phase", p)
        # # Press 'q' to exit the loop

        # # fourier & camera
        # # overlay_rescaled = np.array(overlay, dtype=np.float32)
        # # overlay_rescaled = cv2.resize(overlay_rescaled, (500,500))
        # # overlay_colored = cv2.applyColorMap(overlay_rescaled, cv2.COLORMAP_TWILIGHT_SHIFTED)
        # # cv2.imshow("fourier & camera", overlay_rescaled)
        # ## Ensure overlay_rescaled is a NumPy array in float32 format
        # overlay_rescaled = np.array(overlay_rescaled, dtype=np.float32)
        # # Normalize values between 0 and 255
        # overlay_normalized = cv2.normalize(overlay_rescaled, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert to uint8 for OpenCV compatibility
        # overlay_display = np.clip(overlay_normalized, 0, 255).astype(np.uint8)
        # # Apply a colormap (e.g., COLORMAP_SUMMER for warm tones)
        # overlay_colored = cv2.applyColorMap(overlay_display, cv2.COLORMAP_SUMMER)
        # # Display the pseudo-colored overlay
        # overlay_rescaled_colored = cv2.resize(overlay_colored, (500,500))
        # # cv2.imshow("Fourier Colored Overlay", overlay_rescaled_colored)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Cleanup before exiting
    cv2.destroyAllWindows()
    destroy_hamamatsu(cam_ctx)
    print("✅ Cleanup complete. Camera safely released.")

