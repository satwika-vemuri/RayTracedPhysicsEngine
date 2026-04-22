import cv2
import glob
import os
import re

def hex_key(path):
    filename = os.path.basename(path)
    match = re.search(r'([0-9a-fA-F]+)\.ppm$', filename)
    if match:
        return int(match.group(1), 16)
    return 0


def ppm_to_video(input_folder, output_name='output.avi', fps=24):
    images = glob.glob(os.path.join(input_folder, '*.ppm'))
    images.sort(key=hex_key)
    
    if not images:
        print("No PPM files found.")
        return

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {output_name}")

# usage
ppm_to_video('frames/')
