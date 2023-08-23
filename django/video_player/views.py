from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import video_player
from django.core.files.storage import FileSystemStorage
import sys
sys.path.append('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\CNN-LSTM-Model')
import preprocessing, model, video_search

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

        # Get the URL of the uploaded video for rendering
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
    return render(request, 'process_video.html', {'vid_name':video_url[7:],'frames': frames, 'faceframes': faceframes, 'fps': fps, 'total_frames': total_frames, 'captured_face_frames': len(frames)})

def do_process_video(request, vid_url):
    # video_url = request.GET.get('vid_url')
    print("video_url: ", vid_url)
    video_url = "media/"+vid_url
    frames, faceframes, (fps, total_frames) = preprocessing.extract_faces_from_video('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\\'+video_url,'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\django\process\processed_vid', output_vid=True)
    # return JsonResponse({'result': 'success', 'message': 'Video processed successfully!', 'frames': frames, 'faceframes': faceframes, 'fps': fps, 'total_frames': total_frames, 'captured_frames': len(frames)})
    return JsonResponse({'frames': frames, 'faceframes': faceframes, 'vid_name':video_url[7:], 'fps': fps, 'total_frames': total_frames, 'captured_face_frames': len(faceframes)})
    

def predict_video(request, vid_name):
    # Your processing logic here
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
    return JsonResponse({'result': result, 'prediction': prediction.tolist(), 'pred_mean': str(prediction.mean())})

def search_db(request):
