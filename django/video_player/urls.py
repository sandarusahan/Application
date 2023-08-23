from django.urls import path
from . import views

#URLConf
urlpatterns = [
    path('new/', views.home),
    path('process-video/', views.process_video),
    path('do_process-video/<vid_url>/', views.do_process_video),
    path('predict-video/<vid_name>/', views.predict_video),
    path('search_db/', views.search_db),

]