import cv2
import os
import face_recognition
import glob
import base64

def extract_faces_from_video(video_path, output_folder, num_frames=-1, output_vid = False):
    # Load pre-trained Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}, " +
          f"Width: {width}, Height: {height}")
    

    if num_frames == -1:
        num_frames = total_frames
        
    # Process frames and extract faces
    count = 0
    face_frames_b64 = []
    face_frames = []
    
    frames = []
    frames_b64 = []

    while count < num_frames and count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (240, 240))
        ret2, jpeg = cv2.imencode('.jpg', frame)
        jpeg_b64 = str(base64.b64encode(jpeg))[2:-1]
        frames.append(frame)
        frames_b64.append(jpeg_b64)
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop and save the face
            face_crop = frame[y:y+h, x:x+w]
            # output_filename = os.path.join(
            #     output_folder, f"frame_{count:03d}_face.jpg")
            # cv2.imwrite(output_filename, face_crop)

            ret_face, jpeg_face = cv2.imencode('.jpg', face_crop)
            jpeg_face_b64 = str(base64.b64encode(jpeg_face))[2:-1]
            face_frames_b64.append(jpeg_face_b64)
            face_frames.append(face_crop)
            # Display the cropped face (optional)
            # cv2.imshow("Cropped Face", face_crop)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

            count += 1

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {len(face_frames)} face frames")
    # Create a new video from the face frames
    if output_vid:
        save_face_vid(face_frames, video_path, output_folder)

    return frames_b64, face_frames_b64, (fps, total_frames)

def extract_faces_using_face_recognition(video_path, output_folder, num_frames=-1, output_vid = False):
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}, " +
        f"Width: {width}, Height: {height}")

    if num_frames == -1:
        num_frames = total_frames

    # Process frames and extract faces
    count = 0
    face_frames_b64 = []
    face_frames = []
    frames = []
    frames_b64 = []

    while count < num_frames and count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (240, 240))
        ret2, jpeg = cv2.imencode('.jpg', frame)
        jpeg_b64 = str(base64.b64encode(jpeg))[2:-1]
        frames_b64.append(jpeg_b64)
        # Find face locations in the frame
        face_locations = face_recognition.face_locations(frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_crop = frame[top:bottom, left:right]
            
            ret_face, jpeg_face = cv2.imencode('.jpg', face_crop)
            jpeg_face_b64 = str(base64.b64encode(jpeg_face))[2:-1]
            face_frames_b64.append(jpeg_face_b64)
            face_frames.append(face_crop)

        count += 1

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extracted {len(face_frames)} face frames")

    if output_vid:
        save_face_vid(face_frames, video_path, output_folder)
    return frames_b64, face_frames_b64, (fps, total_frames)

def save_face_vid(face_frames, video_path, output_folder):
    if len(face_frames) > 0:
        out_vid_name = video_path.split('\\')[-1].split('.')[0]
        out_vid_name = out_vid_name[7:]
        print(out_vid_name+" out put path: "+output_folder)
        new_video_path = os.path.join(output_folder, out_vid_name+".mp4")
        # Create output folder if not exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(new_video_path)
        height, width, _ = face_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
        out = cv2.VideoWriter(new_video_path, fourcc,
                              float(24), (width, height))
        print(f"Saving face video to {new_video_path} " +
              str(len(face_frames))+" frames")
        for frame in face_frames:
            out.write(frame)

        out.release()
if __name__ == "__main__":
    # Replace with the path to your video file
    # video_path = "../Sample_DeepTomCruise.mp4"
    # dir = 'O:\Documents\MSc\Dissertation\Datasets\Celeb-DF\Celeb-DF\Fake'
    # dir = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\Celeb-DF\Celeb-DF\Fake'
    # dir = "D:\MSc\FF_dataset\manipulated_sequences\Mix"
    dir = "D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\\train_sample_videos\\real"
    files = glob.glob(os.path.join(dir, '*.mp4'))
    # print(files)
    # video_path = "../sample.mp4"
    output_folder = "Output\Real-DFCD"  # Folder to save the extracted faces
    for video_path in files:
        # print(video_path)
        # extract_faces_from_video(
        #     video_path, output_folder, output_vid=True)

        extract_faces_using_face_recognition(
            video_path, output_folder, output_vid=True)
