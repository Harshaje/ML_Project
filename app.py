from flask import Flask, jsonify, render_template, request
import joblib
from  flask_cors import cross_origin
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
@cross_origin()
def index():
    return render_template("home.html")
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def result():
    item_identifire = float(request.form['item_identifire'])
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_identifire = float(request.form['outlet_identifire'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    Item_Fat_Content_LowFat=0
    Item_Fat_Content_Regular=0

    if item_fat_content == "1":
        Item_Fat_Content_LowFat = 1
    else:
        Item_Fat_Content_Regular = 1

    Outlet_Size_High=0
    Outlet_Size_Medium=0
    Outlet_Size_Small=0

    if outlet_size == "0":
        Outlet_Size_High = 1
    elif outlet_size == "1":
        Outlet_Size_Medium = 1
    else:
        Outlet_Size_Small = 1

    Outlet_Location_Type_Tier1=0
    Outlet_Location_Type_Tier2=0
    Outlet_Location_Type_Tier3=0

    if outlet_location_type == "0":
        Outlet_Location_Type_Tier1 = 1
    elif outlet_location_type == "1":
        Outlet_Location_Type_Tier2 = 1
    else:
        Outlet_Location_Type_Tier3 = 1

    Outlet_Type_GroceryStore=0
    Outlet_Type_SupermarketType1=0
    Outlet_Type_SupermarketType2=0
    Outlet_Type_SupermarketType3=0

    if outlet_type == "0":
        Outlet_Type_GroceryStore = 1
    elif outlet_type == "1":
        Outlet_Type_SupermarketType1 = 1
    elif outlet_type == "2":
        Outlet_Type_SupermarketType2 = 1
    else:
        Outlet_Type_SupermarketType3 = 1


    X= np.array([[item_identifire,item_weight,item_type,item_mrp,outlet_identifire,
                  outlet_establishment_year,Item_Fat_Content_LowFat,
    Item_Fat_Content_Regular,Outlet_Size_High,
    Outlet_Size_Medium,Outlet_Size_Small,Outlet_Location_Type_Tier1,
    Outlet_Location_Type_Tier2,Outlet_Location_Type_Tier3,Outlet_Type_GroceryStore,
    Outlet_Type_SupermarketType1,Outlet_Type_SupermarketType2,Outlet_Type_SupermarketType3]])

    scaler_path= r'C:\Users\harshada\Project_BigMart\models\stand.sav'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'C:\Users\harshada\Project_BigMart\models\rand1.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)
    #y_pred1=float(Y_pred)
    return jsonify({'Prediction': float(Y_pred)})
    #return f'Prediction={y_pred1} '
    #with open('predict.html','w')as html_file:
     #   html_file.write({y_pred1})
    #return render_template('home.html',y_pred1)
    #return f'Prediction={y_pred1} '


if __name__ == "__main__":
    app.run(debug=True, port=8080)
