from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import datetime
from datetime import date

global username

def ViewUsers(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th>Username</th><th>Password</th><th>Contact No</th><th>Address</th>'
        output+='<th>Email ID</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from newuser")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                contact = getContact(row[1])
                output+='<td>'+row[0]+'</td><td>'+str(row[1])+'</td><td>'+row[2]+'</td><td>'+row[3]+'</td><td>'+row[4]+'</td></tr>'
        context= {'data':output}
        return render(request, 'AdminScreen.html', context) 

def ViewAdminBook(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th>Book ID</th><th>Seller Name</th><th>Seller Contact No</th><th>Book Name</th>'
        output+='<th>Category</th><th>Author Name</th><th>Subject</th><th>Cost</th>'
        output+='<th>Image</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from addbook")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                contact = getContact(row[1])
                output+='<td>'+row[0]+'</td><td>'+str(row[1])+'</td><td>'+contact+'</td><td>'+row[2]+'</td><td>'+row[3]+'</td><td>'+row[4]+'</td><td>'+row[5]+'</td>'
                output+='<td>'+row[6]+'</td>'
                output+='<td><img src=/static/Books/'+row[7]+' width=200 height=200></img></td></tr>'
        context= {'data':output}
        return render(request, 'AdminScreen.html', context) 

def ViewTransactions(request):
    if request.method == 'GET':
        global username
        output = '<table border=1><tr><th><font size="" color=black>Order ID</font></th>'
        output+='<td><font size="" color="black">Buyer Name</td>'
        output+='<td><font size="" color="black">Seller Name</td>'
        output+='<td><font size="" color="black">Card No</td>'
        output+='<td><font size="" color="black">CVV No</td>'
        output+='<td><font size="" color="black">Expiry Date</td>'
        output+='<td><font size="" color="black">Ordered Date</td>'
        output+='<td><font size="" color="black">Purchased Book ID</td>'
        output+='<td><font size="" color="black">Total Amount</td></tr>'
        rank = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from orders")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+str(row[0])+'</td>'
                output+='<td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+str(row[3])+'</td>'
                output+='<td><font size="" color="black">'+str(row[4])+'</td>'
                output+='<td><font size="" color="black">'+str(row[5])+'</td>'
                output+='<td><font size="" color="black">'+str(row[6])+'</td>'
                output+='<td><font size="" color="black">'+str(row[7])+'</td>'
                output+='<td><font size="" color="black">'+str(row[8])+'</td></tr>'
        output += "</table><br/><br/><br/>"
        context= {'data': output}
        return render(request, 'AdminScreen.html', context)

def Sold(request):
    if request.method == 'GET':
        global username
        output = '<table border=1><tr><th><font size="" color=black>Order ID</font></th>'
        output+='<td><font size="" color="black">Buyer Name</td>'
        output+='<td><font size="" color="black">Seller Name</td>'
        output+='<td><font size="" color="black">Card No</td>'
        output+='<td><font size="" color="black">CVV No</td>'
        output+='<td><font size="" color="black">Expiry Date</td>'
        output+='<td><font size="" color="black">Ordered Date</td>'
        output+='<td><font size="" color="black">Purchased Book ID</td>'
        output+='<td><font size="" color="black">Total Amount</td></tr>'
        rank = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from orders where seller='"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+str(row[0])+'</td>'
                output+='<td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+str(row[3])+'</td>'
                output+='<td><font size="" color="black">'+str(row[4])+'</td>'
                output+='<td><font size="" color="black">'+str(row[5])+'</td>'
                output+='<td><font size="" color="black">'+str(row[6])+'</td>'
                output+='<td><font size="" color="black">'+str(row[7])+'</td>'
                output+='<td><font size="" color="black">'+str(row[8])+'</td></tr>'
        output += "</table><br/><br/><br/>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def Orders(request):
    if request.method == 'GET':
        global username
        output = '<table border=1><tr><th><font size="" color=black>Order ID</font></th>'
        output+='<td><font size="" color="black">Buyer Name</td>'
        output+='<td><font size="" color="black">Seller Name</td>'
        output+='<td><font size="" color="black">Card No</td>'
        output+='<td><font size="" color="black">CVV No</td>'
        output+='<td><font size="" color="black">Expiry Date</td>'
        output+='<td><font size="" color="black">Ordered Date</td>'
        output+='<td><font size="" color="black">Purchased Book ID</td>'
        output+='<td><font size="" color="black">Total Amount</td></tr>'
        rank = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from orders where buyer='"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="" color="black">'+str(row[0])+'</td>'
                output+='<td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+str(row[2])+'</td>'
                output+='<td><font size="" color="black">'+str(row[3])+'</td>'
                output+='<td><font size="" color="black">'+str(row[4])+'</td>'
                output+='<td><font size="" color="black">'+str(row[5])+'</td>'
                output+='<td><font size="" color="black">'+str(row[6])+'</td>'
                output+='<td><font size="" color="black">'+str(row[7])+'</td>'
                output+='<td><font size="" color="black">'+str(row[8])+'</td></tr>'
        output += "</table><br/><br/><br/>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def getSeller(book_id):
    seller_name = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select seller_name FROM addbook where book_id='"+book_id+"'")
        rows = cur.fetchall()
        for row in rows:
            seller_name = row[0]
            break
    return seller_name

def PurchaseAction(request):
    if request.method == 'POST':
        global username
        buyer = request.POST.get('t1', False)
        book = request.POST.get('t2', False)
        cost = request.POST.get('t3', False)
        card = request.POST.get('t4', False)
        cvv = request.POST.get('t5', False)
        year = request.POST.get('t6', False)
        month = request.POST.get('t7', False)
        seller = getSeller(book)
        order_id = 0
        con = pymysql.connect(host='127.0.0.1', port = 3306, user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select max(order_id) from orders")
            rows = cur.fetchall()
            for row in rows:
                order_id = row[0]
        if order_id is not None:
            order_id = order_id + 1
        else:
            order_id = 1
        today = date.today()
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO orders(order_id,buyer,seller,card_no,cvv_no,expiry_date,order_date,product_id,total_amount) VALUES('"+str(order_id)+"','"+buyer+"','"+seller+"','"+card+"','"+cvv+"','"+month+"-"+year+"','"+str(today)+"','"+book+"','"+str(cost)+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            status = 'Payment Completed Successfully<br/>Total Amount : '+str(cost)
        context= {'data':status}
        return render(request, 'UserScreen.html', context)

def getContact(user):
    contact_no = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select contact_no FROM newuser where username='"+user+"'")
        rows = cur.fetchall()
        for row in rows:
            contact_no = row[0]
            break
    return contact_no

def Purchase(request):
    if request.method == 'GET':
        global username
        pid = request.GET['pid']
        cost = request.GET['cost']
        output = '<tr><td><font size="3" color="black">Buyer&nbsp;Name</b></td><td><input type="text" name="t1" size="40" value="'+username+'" readonly/></td></tr>'
        output +='<tr><td><font size="3" color="black">Purchasing&nbsp;Book&nbsp;ID</b></td><td><input type="text" name="t2" size="20" value="'+pid+'" readonly/></td></tr>'
        output +='<tr><td><font size="3" color="black">Cost</b></td><td><input type="text" name="t3" size="20" value="'+cost+'" readonly/></td></tr>'
        output +='<tr><td><font size="3" color="black">Card&nbsp;No</b></td><td><input type="text" name="t4" size="40" /></td></tr>'
        output +='<tr><td><font size="3" color="black">CVV&nbsp;No</b></td><td><input type="text" name="t5" size="15" /></td></tr>'
        output +='<tr><td><b>Expiry&nbsp;Date</b></td><td><select name="t6">'
        for i in range(2023, 2050):
            output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += '</select>&nbsp;<select name="t7">'
        for i in range(1, 13):
            output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += "</select></td></tr>"
        context= {'data1':output}
        return render(request, 'Purchase.html', context) 

def BookList(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th>Book ID</th><th>Seller Name</th><th>Seller Contact No</th><th>Book Name</th>'
        output+='<th>Category</th><th>Author Name</th><th>Subject</th><th>Cost</th>'
        output+='<th>Image</th><th>Buy Book</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from addbook")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                contact = getContact(row[1])
                output+='<td>'+row[0]+'</td><td>'+str(row[1])+'</td><td>'+contact+'</td><td>'+row[2]+'</td><td>'+row[3]+'</td><td>'+row[4]+'</td><td>'+row[5]+'</td>'
                output+='<td>'+row[6]+'</td>'
                output+='<td><img src=/static/Books/'+row[7]+' width=200 height=200></img></td><td><a href=\'Purchase?pid='+row[0]+'&cost='+row[6]+'\'>Click Here</a></td></tr>'
        print(output)        
        context= {'data':output}
        return render(request, 'UserScreen.html', context)  

def AddBooksAction(request):
    if request.method == 'POST':
        global username
        name = request.POST.get('t1', False)
        category = request.POST.get('t2', False)
        author = request.POST.get('t3', False)
        subject = request.POST.get('t4', False)
        cost = request.POST.get('t5', False)
        book = request.FILES['t6']
        file_id = 0
        con = pymysql.connect(host='127.0.0.1', port = 3306, user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select max(book_id) from addbook")
            rows = cur.fetchall()
            for row in rows:
                file_id = row[0]
        if file_id is not None:
            file_id = int(file_id) + 1
        else:
            file_id = 1
        fs = FileSystemStorage()
        filename = fs.save('BookApp/static/Books/'+str(file_id)+'.png', book)
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO addbook(book_id,seller_name,book_name,category,author_names,subject,cost,image) VALUES('"+str(file_id)+"','"+username+"','"+name+"','"+category+"','"+author+"','"+subject+"','"+cost+"','"+str(file_id)+'.png'+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            context= {'data':'Book Details Added with Book ID : '+str(file_id)}
            return render(request, 'AddBooks.html', context)
        else:
            context= {'data':'Error in adding book details'}
            return render(request, 'AddBooks.html', context)    

def AddBooks(request):
    if request.method == 'GET':
        return render(request, 'AddBooks.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})    

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})   

def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM newuser where username='"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                status = "exists"
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO newuser(username,password,contact_no,address,email) VALUES('"+username+"','"+password+"','"+contact+"','"+address+"','"+email+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                context= {'data':'Signup Process Completed'}
                return render(request, 'Register.html', context)
            else:
                context= {'data':'Error in signup process'}
                return render(request, 'Register.html', context)
        else:
            context= {'data':'Username already exists'}
            return render(request, 'Register.html', context) 
        
def UserLogin(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        utype = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'BookResale',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM newuser")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    utype = "success"
                    break
        if utype == 'success':
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        if utype == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)        
        
        
def AdminLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)


        
