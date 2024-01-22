from django.shortcuts import render

from .LoadModel import test_model1, test_model2

# These models are not imported since the the LLMS are not used in this portable version of the application
# , test_model3, test_model4


# Here are the definitions of the functions (views) that are called in the LoadModel.py file

def index(request):
    result = None

    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        result = test_model1(input_data)  # Call the Python script function

    return render(request, 'mainUI/index.html', {'result': result})

def lstm2(request):
    result = None

    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        result = test_model2(input_data)  

    return render(request, 'mainUI/lstm2.html', {'result': result})

def mistral(request):
    result = None

    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        result = test_model3(input_data)  

    return render(request, 'mainUI/mistral.html', {'result': result})

def wizard(request):
    result = None

    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        result = test_model4(input_data)  

    return render(request, 'mainUI/wizard.html', {'result': result})