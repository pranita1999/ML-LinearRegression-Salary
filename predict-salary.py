import joblib
salary = joblib.load("SalaryModel.pkl")
x=int(input("Enter your Experience")
print(salary.predict([[x]])
      
      #This file uses the model to predict the salary. The model is loaded and the value is predicted.
