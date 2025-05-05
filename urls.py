from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('Login.html', views.Login, name="Login"), 
	       path('AdminLogin.html', views.AdminLogin, name="AdminLogin"), 
	       path('Register.html', views.Register, name="Register"),
	       path('Signup', views.Signup, name="Signup"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('AdminLoginAction', views.AdminLoginAction, name="AdminLoginAction"),
	       path('AddBooks.html', views.AddBooks, name="AddBooks"),
	       path('AddBooksAction', views.AddBooksAction, name="AddBooksAction"),
	       path('BookList', views.BookList, name="BookList"),
	       path('Purchase', views.Purchase, name="Purchase"),
	       path('PurchaseAction', views.PurchaseAction, name="PurchaseAction"),
	       path('Orders', views.Orders, name="Orders"),
	       path('Sold', views.Sold, name="Sold"),
	       path('ViewAdminBook', views.ViewAdminBook, name="ViewAdminBook"),
	       path('ViewUsers', views.ViewUsers, name="ViewUsers"),
	       path('ViewTransactions', views.ViewTransactions, name="ViewTransactions"),
]