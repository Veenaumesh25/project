from django.shortcuts import render

# Create your views here.

def register(request):
    if(request.method=="POST"):
        data=request.POST #store all the values of the textboxes
        firstname=data.get('textfirstname') #by the name of the textbox
        lastname=data.get('textlastname')
        if('buttonsubmit' in request.POST):
            result=firstname+lastname
            return render(request,'register.html',context={'result':result})
    return render(request,'register.html')

def employee(request):
    if(request.method=="POST"):
        data=request.POST
        emailid=data.get('textemailid')
        mobileno=data.get('textmobileno')
        if('buttonsubmit' in request.POST):
            result=emailid+mobileno
            return render(request,'employee.html',context={'result':result})
    return render(request,'employee.html')

def calci(request):
    if(request.method=="POST"):
        data=request.POST
        firstnumber=data.get('textfirstnumber')
        secondnumber=data.get('textsecondnumber')
        if('buttonadd' in request.POST):
            result=int(firstnumber)+int(secondnumber)
            return render(request,'calci.html',context={'result':result})
        if('buttonsub' in request.POST):
            result=int(firstnumber)-int(secondnumber)
            return render(request,'calci.html',context={'result':result})
        if('buttonmul' in request.POST):
            result=int(firstnumber)*int(secondnumber)
            return render(request,'calci.html',context={'result':result})
        if('buttondiv' in request.POST):
            result=int(firstnumber)/int(secondnumber)
            return render(request,'calci.html',context={'result':result})
    return render(request,'calci.html')

def index(request):
    return render(request,'index.html')


def marks(request):
    if(request.method=="POST"):
        data=request.POST
        hours=data.get('textmarks')
        age=data.get('textage')
        internet=data.get('textinternet')
        if('buttonpredict' in request.POST):
            import pandas as pd
            path="C:/Users/veena umesh/OneDrive/Desktop/internship/Data/Exammarks.csv"
            data=pd.read_csv(path)
            medianvalue=data.hours.median()
            
            data.hours=data.hours.fillna(medianvalue)
           
            inputs=data.drop('marks',axis=1)
            output=data.drop(['hours','age','internet'],axis=1)
            import sklearn
            import math
            from sklearn import linear_model
            model=linear_model.LinearRegression()
            model.fit(inputs,output)
            result=model.predict([[float(hours),int(age),int(internet)]])
            return render(request,'marks.html',context={'result':result})


def dryb(request):
    if(request.method=="POST"):
        data=request.POST
        area=data.get('textarea')
        perimeter=data.get('textperimeter')
        majoraxis=data.get('textmajoraxis')
        minoraxis=data.get('textminoraxis')
        aspectration=data.get('textaspectration')
        eccentricity=data.get('texteccentricity')
        convexarea=data.get('textconvexarea')
        equivdiameter=data.get('textequivdiameter')
        extent=data.get('textextent')
        solidity=data.get('textsolidity')
        roundness=data.get('textroundness')
        compactness=data.get('textcompactness')
        shapefactor1=data.get('textshapefactora')
        shapefactor2=data.get('textshapefactorb')
        shapefactor3=data.get('textshapefactorc')
        shapefactor4=data.get('textshapefactord')
        if('buttonpredict' in request.POST):
            import pandas as pd
            path="C:/Users/veena umesh/OneDrive/Desktop/internship/train_dataset.csv"
            data=pd.read_csv(path)

            inputs=data.drop(['Class'],'columns')
            output=data['Class']

            import sklearn
            from sklearn.model_selection import train_test_split
            x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)

            from sklearn.neighbors import KNeighborsClassifier
            model=KNeighborsClassifier(n_neighbors=13)
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            

            result=model.predict([[int(area or 0),float(perimeter or 0),float(majoraxis or 0),float(minoraxis or 0),float(aspectration or 0),float(eccentricity or 0),int(convexarea or 0),float(equivdiameter or 0),float(extent or 0),float(solidity or 0),float(roundness or 0),float(compactness or 0),float(shapefactor1 or 0),float(shapefactor2 or 0),float(shapefactor3 or 0),float(shapefactor4 or 0)]])
            #print(res)
            return render(request,'dryb.html',context={'result':result})
    return render(request,'dryb.html')

def rice(request):
    if(request.method=="POST"):
        data=request.POST
        area=data.get('textarea')
        majoraxislength=data.get('textmajoraxis')
        minoraxislength=data.get('textminoraxis')
        eccentricity=data.get('texteccentricity')
        convexarea=data.get('textconvexarea')
        equivdiameter=data.get('textequivdiameter')
        extent=data.get('textextent')
        perimeter=data.get('textperimeter')
        roundness=data.get('textroundness')
        aspectration=data.get('textaspectration')
        if('buttonpredict' in request.POST):
            import pandas as pd
            path="C:/Users/veena umesh/OneDrive/Desktop/internship/riceClassification.csv"
            data=pd.read_csv(path)
            
            inputs=data.drop(['id','Class'],axis=1)
            output=data['Class']
            
            import sklearn
            from sklearn.model_selection import train_test_split
            x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
            

            import sklearn
            from sklearn.ensemble import RandomForestClassifier
            model=RandomForestClassifier(n_estimators=50)
            model.fit(x_train,y_train)

            y_pred=model.predict(x_test)
            
            result=model.predict([[int(area or 0),float(majoraxislength or 0),float(minoraxislength or 0),float(eccentricity or 0),float(convexarea or 0),float(equivdiameter or 0),float(extent or 0),float(perimeter or 0),float(roundness or 0),float(aspectration or 0)]])
            return render(request,'rice.html',context={'result':result})
    return render(request,'rice.html')