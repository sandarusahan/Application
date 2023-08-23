import os
from imagehash import phash
from PIL import Image
import cv2
import json
import distance

# Function to calculate pHash for a video frame
def calculate_phash(vid_path, vid_res=(128,128)):
    cap = cv2.VideoCapture(vid_path)
    frames_hashes = []
    if not cap.isOpened():
        print("Error opening video file.")
        return
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_capture = [0, int(vid_len*0.25), int(vid_len*0.5), int(vid_len*0.75), vid_len]
    for i in range(len(frame_capture)):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_capture:
            frame = format_frame(frame, vid_res)
            frames_hashes.append(phash(Image.fromarray(frame)).__str__())
        
    return frames_hashes

def format_frame(frame, output_size):
    # geyscale image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, output_size)
    return frame

def hamming_distance(hash1, hash2):
    return distance.hamming(hash1, hash2)

def create_metadata_json(directory_path):
    # Create a CSV file to store the pHash of all the videos
    vid_metadata_list = []
    
    # Iterate through the directory and calculate the pHash of all the videos
    vid_id = 0
    for root, dirs, files in os.walk(directory_path):
        print(files)
        for file in files:
            if file.endswith('.mp4'):
                vid_meta = VidMeta()
                vid_id += 1
                vid_meta.vid_id = vid_id
                vid_meta.vid_name = file.split('.')[0]
                vid_meta.vid_path = os.path.join(root, file)
                vid_meta.vid_hash = calculate_phash(vid_meta.vid_path)
                vid_meta.vid_label = root.split('\\')[-1]
                vid_metadata_list.append(vid_meta)
                
    print(len(vid_metadata_list))
    vid_metadata_json = json.dumps([ob.__dict__ for ob in vid_metadata_list])
    # Write the metadata to a JSON file
    with open('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\known_videos\metadata.json', 'w') as f:
        f.write(vid_metadata_json)
def find_hashes_in_dict(vid_meta_dict):
    hashes_n_id = []
    id = 0
    hashes = []
    for key in vid_meta_dict:
        if key == 'vid_id':
            print(vid_meta_dict[key])
            id = vid_meta_dict[key]
        if key == 'vid_hash':
            print(vid_meta_dict[key])
            hashes = vid_meta_dict[key]
        hashes_n_id.append([id, [h for h in hashes]])
    return hashes_n_id
# Function to check if a video is available in a directory
def find_simillar_vids(video_path, acc_thresh=5):
    results = []
    video_hash = calculate_phash(video_path)
    # vid_id, vid_name, vid_hash, vid_path, vid_label
    # Read the CSV file containing the pHash of all the videos
    vid_metadata_path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\known_videos\metadata.json'
    with open (vid_metadata_path, 'r') as f:
        vid_metadata_dict = json.loads(f.read())

    # Iterate through the dictionary and find the video with the same pHash
    for vid_meta in vid_metadata_dict:
        found_vid_meta = VidMeta() 
        for src_hash in list(vid_meta['vid_hash']):
            for tar_hash in list(video_hash): 
                if hamming_distance(src_hash, tar_hash) <= acc_thresh:
                    found_vid_meta.vid_id = vid_meta['vid_id']
                    found_vid_meta.vid_name = vid_meta['vid_name']
                    found_vid_meta.vid_hash = vid_meta['vid_hash']
                    found_vid_meta.vid_path = vid_meta['vid_path']
                    found_vid_meta.vid_label = vid_meta['vid_label']
                    if(found_vid_meta.vid_id not in [r.vid_id for r in results]):
                        results.append(found_vid_meta)
                

    return results
   
class VidMeta:
    def __init__(self, vid_id, vid_name, vid_hash, vid_path, vid_label):
        self.vid_id = vid_id
        self.vid_name = vid_name
        self.vid_hash = vid_hash
        self.vid_path = vid_path
        self.vid_label = vid_label
    
    def __init__(self):
        self.vid_id = 0
        self.vid_name = ''
        self.vid_hash = ''
        self.vid_path = ''
        self.vid_label = ''

    def __str__(self):
        return f'Video ID: {self.vid_id}, Video Name: {self.vid_name}, Video Hash: {self.vid_hash}, Video Path: {self.vid_path}, Video Label: {self.vid_label}'
    

# if __name__ == "__main__":
#     # create_metadata_json('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\Output\Fake_Sample')
#     sim_vids = find_simillar_vids('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model\Output\Fake_Sample\id0_id3_0008.mp4')
#     print([v.vid_name for v in sim_vids])
