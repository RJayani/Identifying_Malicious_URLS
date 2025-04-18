# Libraries
from django.shortcuts import render,redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from fakenews.models import User

################ Home #################
def home(request):
	return render(request,'home1.html')
def login(request):
	return render(request,'loginform.html')
def loginCheck(request):
	if request.method == 'POST':
		firstname = request.POST.get('username')
		password = request.POST.get('email')
		request.session['firstname'] = firstname 
		print(firstname)
		print(password)
		user_object=User.objects.get(firstname=firstname,password=password)
		print('--------------------')
		print(user_object)
		try:
			print('hi')
			user_object=User.objects.get(firstname=firstname,password=password)
			print(user_object)
			if user_object is not None:
				print('hiiiiiiii')
				request.session['useremail'] = user_object.email
				return redirect('home')
		except:
			#user_object = None
			print('hello')
			return redirect('login')
	return render(request,'home.html')	
def logout(request):
	return render(request,'index.html')	
def reg(request):
	return render(request,'register.html')

######## SVM ######
def save(request):
	if request.method == 'POST':
		print('printtttttttttttttttttttttttttttttttt')
		print('checkkkkkkkkkkkkkkkkk')
		username= request.POST.get('username')
		password= request.POST.get('password')
		address= request.POST.get('address')
		email= request.POST.get('email')
		age= request.POST.get('age')
		gender= request.POST.get('gender')
		phone= request.POST.get('phone')
		user=User()
		user.firstname= request.POST.get('username')
		user.password= request.POST.get('password')
		user.address= request.POST.get('address')
		user.email= request.POST.get('email')
		user.age= request.POST.get('age')
		user.gender= request.POST.get('gender')
		user.phone= request.POST.get('phone')
		user.save()		
		return render(request,'loginform.html')
	return render(request,'loginform.html')	


######## SVM ######
def nvb(request):
	return render(request,'pacweb1.html')
def pac(request):
	if request.method == 'POST':
		if request.method == 'POST':
			headline1= request.POST.get('headline1')
			headline1= headline1
			atest=[headline1]
			from django.shortcuts import render
			from django.http import HttpResponse
			import pandas as pd
			import numpy as np
			import matplotlib.pyplot as plt
			from sklearn.model_selection import train_test_split
			from sklearn.feature_extraction.text import TfidfVectorizer
			import itertools
			from sklearn import metrics
			import os
			import seaborn as sns
			from sklearn.model_selection import train_test_split
			from sklearn.metrics import confusion_matrix
			df = pd.read_excel('C:/Users/R Jayani/Downloads/Major Project - Copy/phishing web site/phishingwebsite.xlsx')
			y = df.label
			X = df.Domain=df.Domain.astype(str)
			print(X.shape)
			print(y.shape)
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
			tfidf_vect = TfidfVectorizer(stop_words = 'english')
			tfidf_train = tfidf_vect.fit_transform(X_train)
			tfidf_test = tfidf_vect.transform(X_test)
			atest = tfidf_vect.transform(atest)

			tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
			
			from sklearn.linear_model import PassiveAggressiveClassifier
			linear_clf= PassiveAggressiveClassifier()
			linear_clf.fit(tfidf_train, y_train)
			pred = linear_clf.predict(tfidf_test)
			pred1 = linear_clf.predict(atest)
			print(pred1)
			print(pred)
			value=''
			if pred1 <= 0:	  
				value = "Legitimate"
			else:
				value = "Malicious"
			score = metrics.accuracy_score(y_test, pred)
			d={'predictedvalue':value,'accuracy':score}				 
	return render(request,'result.html',d)
def svm(request):	
	return render(request,'acc1.html')		
def dec(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df =  pd.read_excel('C:/Users/R Jayani/Downloads/Major Project - Copy/phishing web site/phishingwebsite.xlsx')
	y = df.label
	X = df.Domain=df.Domain.astype(str)
	print(X.shape)
	print(y.shape)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	tfidf_vect = TfidfVectorizer(stop_words = 'english')
	tfidf_train = tfidf_vect.fit_transform(X_train)
	tfidf_test = tfidf_vect.transform(X_test)

	tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
			
	from sklearn.tree import DecisionTreeClassifier
	linear_clf= DecisionTreeClassifier(criterion = 'entropy', random_state = 11).fit(tfidf_train, y_train)
	pred = linear_clf.predict(tfidf_test)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	d={'accuracy':score}
	return render(request,'acc1.html',d)
def randomf(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df =  pd.read_excel('C:/Users/R Jayani/Downloads/Major Project - Copy/phishing web site/phishingwebsite.xlsx')
	y = df.label
	X = df.Domain=df.Domain.astype(str)
	print(X.shape)
	print(y.shape)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	tfidf_vect = TfidfVectorizer(stop_words = 'english')
	tfidf_train = tfidf_vect.fit_transform(X_train)
	tfidf_test = tfidf_vect.transform(X_test)

	tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
	linear_clf= MultinomialNB()
	linear_clf.fit(tfidf_train, y_train)
	pred = linear_clf.predict(tfidf_test)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	d={'accuracy':score}
	return render(request,'acc1.html',d)
def mnb(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df =  pd.read_excel('C:/Users/R Jayani/Downloads/Major Project - Copy/phishing web site/phishingwebsite.xlsx')
	y = df.label
	X = df.Domain=df.Domain.astype(str)
	print(X.shape)
	print(y.shape)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	tfidf_vect = TfidfVectorizer(stop_words = 'english')
	tfidf_train = tfidf_vect.fit_transform(X_train)
	tfidf_test = tfidf_vect.transform(X_test)

	tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
			
	from sklearn.linear_model import PassiveAggressiveClassifier
	linear_clf= SVC()
	linear_clf.fit(tfidf_train, y_train)
	pred = linear_clf.predict(tfidf_test)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	d={'accuracy':score}	
	return render(request,'acc1.html',d)
def graph(request):
	from django.shortcuts import render
	from django.http import HttpResponse
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import TfidfVectorizer
	import itertools
	from sklearn import metrics
	import os
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix
	df = pd.read_excel('C:/Users/R Jayani/Downloads/Major Project - Copy/phishing web site/phishingwebsite.xlsx')
	y = df.label
	X = df.Domain=df.Domain.astype(str)
	print(X.shape)
	print(y.shape)
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
	tfidf_vect = TfidfVectorizer(stop_words = 'english')
	tfidf_train = tfidf_vect.fit_transform(X_train)
	tfidf_test = tfidf_vect.transform(X_test)

	tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
			
	from sklearn.linear_model import PassiveAggressiveClassifier
	linear_clf= KNeighborsClassifier()
	linear_clf.fit(tfidf_train, y_train)
	pred = linear_clf.predict(tfidf_test)
	print(pred)
	score = metrics.accuracy_score(y_test, pred)
	d={'accuracy':score}
	return render(request,'acc1.html',d)	
def accuracy(request):
	return render(request,'index.html')
			