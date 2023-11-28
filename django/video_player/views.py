import os
import subprocess
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import video_player
from django.core.files.storage import FileSystemStorage
import sys
sys.path.append('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model')
import preprocessing, model, video_search
import json

def blank(request):
    return redirect('home')

def home(request):
    if request.method == 'GET':
        da = video_player.objects.all()
        # return HttpResponse('Hello, world.')
        return render(request, 'home.html', {'da':da})


    if request.method == 'POST' and request.FILES.get('video_file'):
        global video_url
        video_file = request.FILES['video_file']

        # Save the uploaded video to a specific location
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)

        video_url = fs.url(filename)
        return render(request, 'home.html', {'video_url': video_url})

    
    return render(request, 'home.html')

def process_video(request):
    # Your processing logic here
    # preprocessing.main()
    frames, faceframes = [], []
    video_url = request.GET.get('video_url')
    fps, total_frames = 0, 0
    frames, faceframes, (fps, total_frames) = preprocessing.extract_faces_from_video('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\\'+video_url,'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid', output_vid=True)

    # print(request.GET.get('video_url'))
    # print(frames[0])
    return render(request, 'process_video.html', {'vid_name':video_url[7:],'frames': frames, 'faceframes': faceframes, 'fps': fps, 'total_frames': total_frames, 'captured_face_frames': len(faceframes)})

def do_process_video(request, vid_url):
    # video_url = request.GET.get('vid_url')
    print("video_url: ", vid_url)
    video_url = "media/"+vid_url
    frames, faceframes, (fps, total_frames) = preprocessing.extract_faces_from_video('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\\'+video_url,'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid', output_vid=True)
    # return JsonResponse({'result': 'success', 'message': 'Video processed successfully!', 'frames': frames, 'faceframes': faceframes, 'fps': fps, 'total_frames': total_frames, 'captured_frames': len(frames)})
    return JsonResponse({'frames': frames, 'faceframes': faceframes, 'vid_name':video_url[7:], 'fps': fps, 'total_frames': total_frames, 'captured_face_frames': len(faceframes)})
    

def predict_video(request, vid_name):
    path = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid\\'+vid_name
    prediction = model.predict_video(path)
    result = ""
    pred_mean = prediction.mean()
    if pred_mean < 0.3:
        result = "Fake"
    elif pred_mean < 0.4 and pred_mean > 0.3:
        result = "Probably fake"
    elif pred_mean < 0.55 and pred_mean > 0.45:
        result = "Could be real or fake"
    elif pred_mean < 0.7 and pred_mean > 0.55:
        result = "Probably real"
    else:
        result = "Real"
        # show the probability of the prediction
    return JsonResponse({'result': result, 'predictions': prediction.tolist(), 'pred_mean': str(prediction.mean())})

def search_db(request, vid_path):
    vid_path = "media/"+ vid_path
    vid_results = video_search.find_simillar_vids(vid_path)
    message = str(len(vid_results)) + ' Similar video(s) found'
    if not len(vid_results) > 0:
        message = 'No similar videos found'
    
    result = json.dumps(vid_results)
    return JsonResponse({'result': result, 'message': message})

def save_db(request, vid_name, label):
    vid_name = "media/"+ vid_name
    video_search.add_video_to_meta_json(vid_name, label)
    return JsonResponse({'result': 'success', 'message': 'Video saved to database successfully!'})
    
def show_db(request):
    vid_results = video_search.get_all_videos_from_meta_json()
    message = str(len(vid_results)) + ' video(s) found'
    if not len(vid_results) > 0:
        message = 'No videos found'
    # print(vid_results)
    # result = JsonResponse({'result': result, 'message': message})
    return render(request, 'show_db.html', {'result': vid_results, 'message': message})

def play_vid(request, vid_id):
    vid_results = video_search.get_all_videos_from_meta_json()
    vid_path = ""
    for i in range(len(vid_results)):
        if vid_results[i]['vid_id'] == int(vid_id):
            vid_path = vid_results[i]['vid_path']
            if vid_results[i]['vid_label']=='1':
                vid_label = "This video is identified as Real"
            else:
                vid_label = "This video is identified as Fake"
    return render(request, 'play_video.html', {'vid_path': vid_path, 'vid_label': vid_label})