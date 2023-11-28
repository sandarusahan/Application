from django.shortcuts import render, redirect

def blank(request):
    return redirect('video/new/')