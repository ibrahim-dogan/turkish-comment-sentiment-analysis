import os
import pickle

from django.shortcuts import render
# Create your views here.
from django.views.decorators.clickjacking import xframe_options_exempt


def index(request):
    if request.method == 'POST':
        comment_to_predict = request.POST["comment"]
        prediction = mlp_predict(comment_to_predict)

        return render(request, 'index.html', {"comment_to_predict": comment_to_predict, "prediction": prediction})

    return render(request, 'index.html')


def codes(request):
    return render(request, 'report.html')


@xframe_options_exempt
def ipynb(request):
    return render(request, 'ipynb.html')


mlp = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/mlpclassifier.pkl", 'rb'))
vectorizer = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + "/vectorizer.pkl", 'rb'))


def mlp_predict(comment, prediction={}):

    result = mlp.predict(vectorizer.transform([comment]).toarray())
    if result == 1:
        prediction["state"] = "negative"
    elif result == 3:
        prediction["state"] = "positive"
    else:
        prediction["state"] = "agnostic"
    return prediction
