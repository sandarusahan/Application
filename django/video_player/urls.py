from django.urls import path
from django.shortcuts import redirect
from . import views

#URLConf
urlpatterns = [
    path('new/', views.home),
    path('process-video/', views.process_video),
    path('do-process-video/<vid_url>/', views.do_process_video),
    path('predict-video/<vid_name>/', views.predict_video),
    path('search-db/<vid_path>/', views.search_db),
    path('save-db/<vid_name>/<label>/', views.save_db),
    path('show-db/', views.show_db),
    path('show-db/<vid_id>/', views.play_vid),

]