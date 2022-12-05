from django.shortcuts import render, redirect
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage


# def home(request):
#    return render(request, 'interface/home.html')


def result(request):
    return render(request, 'interface/result.html')


def train(request):
    return render(request, 'interface/train.html')


def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'interface/result.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'interface/home.html')
