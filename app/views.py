from django.shortcuts import render

# Create your views here.
from app.svm import predict_row


def index(request):
    if request.method == 'POST':
        print(request.POST)
        row = {'Gender': request.POST['Gender'],
               'Married': request.POST['Married'],
               'Dependents': int(request.POST['Dependents']),
               'Education': request.POST['Education'],
               'Self_Employed': request.POST['Self_Employed'],
               'ApplicantIncome': int(request.POST['ApplicantIncome']),
               'CoapplicantIncome': int(request.POST['CoapplicantIncome']),
               'LoanAmount': int(request.POST['LoanAmount']),
               'Loan_Amount_Term': int(request.POST['Loan_Amount_Term']),
               'Credit_History': int(request.POST['Credit_History']),
               'Property_Area': request.POST['Property_Area']
               }
        prediction = predict_row(row)[0]
        print(prediction)
        if prediction == 1:
            return render(request, 'yes.html')
        elif prediction == 0:
            return render(request, 'no.html')

    return render(request, 'contact-us.html')
