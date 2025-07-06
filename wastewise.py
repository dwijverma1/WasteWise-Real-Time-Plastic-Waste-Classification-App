import os
import streamlit as st
import folium
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import csv
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Dummy user data
users = [
    {"username": "demonsoul", "password": "password", "name": "Dwij"},
    {"username": "diamondguy", "password": "password", "name": "Nikaash"},
    {"username": "notdepressed", "password": "password", "name": "Devyansh"}
]

# Page configuration
st.set_page_config(page_title="WasteWise", layout="wide", page_icon="wastewiselogo.png")
            

# Sidebar title
st.sidebar.title("Login")

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    

    if st.sidebar.button("Login"):
        for user in users:
            if user["username"] == username and user["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.sidebar.success(f"Welcome back, {user['name']}!")
                break
        else:
            st.sidebar.error("Invalid username or password.")

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.sidebar.success("Logged out successfully!")

# Display app content if logged in
if st.session_state.logged_in:
    name = st.session_state.user['name']
    # Sidebar navigation
    st.sidebar.title(f"Welcome {name}")
    with st.sidebar:
        choice = option_menu('Multiple Disease Prediction System',
                            ['HOME',
                            'PLASTIC FOOTPRINT👣',
                            'OBJECT IDENTIFICTION',
                            'PLASTIC SCORE',
                            'INFORMATION',
                            'LIST',
                            'MAP',
                            'DASHBOARD'],
                            menu_icon='',
                            icons=['house', 'cart4', 'camera-video','recycle','info-circle','list-task','map','bar-chart'],
                            default_index=0)

    if choice=="HOME":
        img=Image.open(r'C:\Users\admin\OneDrive\Desktop\wastewise\WW.png')
        new_size=img.resize((500,500))
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(new_size)
            
    # Diabetes Prediction Page
    if choice == "PLASTIC FOOTPRINT👣":
        st.subheader("PLASTIC FOOTPRINT CALCULATOR👣")

        # Define emission factors (example values, replace with accurate data)
        EMISSION_FACTORS = {
            "India": {
                "Food & Kitchen": 195,  # tonnes/day
                "Bathroom & Laundry": 85,  # tonnes/day
                "Disposable Containers & Packaging": 103,  #tonnes/day
                "Others": 15  #tonnes/day
            }
        }

        # User inputs
        st.subheader("🌍 Your Country")
        country = st.selectbox("Select", ["India"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍽️Food & Kitchen (in kg)")
            fd = st.slider("Quantity", 0.0, 100.0, key="quantity1_input")

            st.subheader("🛁Bathroom & Laundry (in kg)")
            bl = st.slider("Quantity", 0.0, 100.0, key="quantity2_input")

        with col2:
            st.subheader("🥤Disposable Containers & Packaging (in kg)")
            waste = st.slider("Quantity", 0.0, 100.0, key="quantitiy3_input")

            st.subheader("Others")
            oth = st.slider("🗑️Quantity", 0.0, 100.0, key="quantitiy4_input")

        # Normalize inputs
        if fd > 0:
            fd = fd * 365  
        if bl > 0:
            bl = bl * 365  
        if oth > 0:
            oth = oth * 365  
        if waste > 0:
            waste = waste * 365  

        
        fd_emissions = EMISSION_FACTORS[country]["Food & Kitchen"] * fd
        bl_emissions = EMISSION_FACTORS[country]["Bathroom & Laundry"] * bl
        oth_emissions = EMISSION_FACTORS[country]["Disposable Containers & Packaging"] * oth
        waste_emissions = EMISSION_FACTORS[country]["Others"] * waste

        # Convert emissions to tonnes and round off to 2 decimal points
        fd_emissions = round(fd_emissions / 1000, 2)
        bl_emissions = round(bl_emissions / 1000, 2)
        oth_emissions = round(oth_emissions / 1000, 2)
        waste_emissions = round(waste_emissions / 1000, 2)

        # Calculate total emissions
        total_emissions = round(
            fd_emissions + bl_emissions + oth_emissions + waste_emissions, 2
        )

        if st.button("Calculate Plastic Footprint👣"):

            # Display results
            st.header("Results")

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Plastic Footprint👣 by Category")
                st.info(f" 🍽️Food & Kitchen Waste: {fd_emissions} tonnes plastic per year")
                st.info(f"🛁 Bathroom & Laundry Waste: {bl_emissions} tonnes plastic per year")
                st.info(f"🥤 Disposable Containers & Packaging: {oth_emissions} tonnes plastic per year")
                st.info(f"🗑️ Waste: {waste_emissions} tonnes plastic per year")

            with col4:
                st.subheader("Total Plastic Footprint👣")
                st.success(f"🌍 Your total plastic footprint👣 is: {total_emissions} tonnes plastic per year")
                st.warning("India's per capita plastic consumption reached 15 kilograms per person in 2021, which is higher than the global average of 20.9 kilograms. Between 1972 and 2021, plastic waste per capita of India grew substantially.")


    elif choice == "OBJECT IDENTIFICTION":
        st.subheader("OBJECT IDENTIFIER")

        cap = cv2.VideoCapture(0)

        # recording
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter("Object_Detection.mp4", fourcc, fps, (width, height))

        # Threshold
        Threshold = 0.5

        classNames= []
        classFile = 'classnames'
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            timer = cv2.getTickCount()
            success,img = cap.read()
            fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
            cv2.putText(img,"fps=",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(127,127,255),2)
            cv2.putText(img,str(int(fps)),(75,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(127,127,255),2)

            classIds, confs, bbox = net.detect(img,confThreshold=Threshold)

            if len(classIds) != 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    print(box,classNames[classId-1].upper(),str(round(confidence*100,2)))

                    x, y, w, h = box
                    top = max(0, np.floor(x + 0.5).astype(int))
                    left = max(0, np.floor(y + 0.5).astype(int))
                    right = min(img.shape[1], np.floor(x + w + 0.5).astype(int))
                    bottom = min(img.shape[0], np.floor(y + h + 0.5).astype(int))

                    # cv2.circle(img, (int((top + right) / 2), int((left + bottom) / 2)), 40, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

                    cv2.rectangle(img, (top-2, left-2), (right+2, bottom+2),color=(238, 255, 0),thickness=1)
                    cv2.rectangle(img, (top-4, left-4), (right+4, bottom+4),color=(255, 145, 0),thickness=1)
                    cv2.rectangle(img, (top-6, left-6), (right+6, bottom+6),color=(255, 0, 111),thickness=1)
                    cv2.rectangle(img, (top-8, left-8), (right+8, bottom+8),color=(255, 0, 238),thickness=1)
                    cv2.rectangle(img, (top-10, left-10), (right+10, bottom+10),color=(145, 0, 255),thickness=1)

                    cv2.rectangle(img,box,color=(111,255,0),thickness=1)

                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+5,box[1]+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),1)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+5,box[1]+40),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)

            writer.write(img)
            cv2.imshow("Object_Detection",img)
            key = cv2.waitKey(1)
            if key == 27:
                    cv2.destroyAllWindows()
                    break
            
    elif choice == "PLASTIC SCORE":
        st.subheader("PLASTIC SCORE")

        check=st.checkbox("COLLECTED")
        check1=st.checkbox("DISPOSED")
        check2=st.checkbox("RECYCLED")

        if check and check1 and check2:
            st.write("Enter your previous Plastic Score: ")
            number = st.number_input('Insert a number',min_value=0,max_value=1000)
            st.write('Your current plastic score is: ', number+5)

    elif choice == "LIST":
        st.subheader("LIST OF WASTE DUMPERS IN MUMBAI")
            
        list=  ["Eco Support Pvt Ltd.",
                "Davidson Facility Services Pvt Ltd",
                "Triton Greentech Innovations Pvt Ltd.",
                "E4f Resurrect Pvt. Ltd. ",
                "Shree Swami Samarth Enviro Consultant Pvt Ltd.",
                "Neem Enviro.",
                "KSV Enterprise"]
        
        for i in list:
            st.markdown("- " + i)
    
    elif choice=="MAP":

        datafile="Electric_Vehicle_Charging_Stations.csv"

        def read_data():
            def parse_lat_lon(point):
                    return point.split("(")[-1].split(")")[0].split()
            
            data=[]
            with open(datafile,'r') as csvfile:
                reader=csv.DictReader(csvfile)
                print("reading...")
                for row in reader:
                    longitude,latitude=parse_lat_lon(row['New Georeferenced Column'])
                    data.append({
                        'name':row['NAME'],
                        'latitude': float(latitude),
                        'longitude': float(longitude)
                    })
                return data
        
        data=read_data()

        CONNECTICUT_CENTER=(19.0760,72.8777)
        map=folium.Map(location=CONNECTICUT_CENTER,zoom_start=9)

        for station in data:
            location=station['latitude'], station['longitude']
            folium.Marker(location,popup=station['name']).add_to(map)

        st.subheader("WASTE DUMPERS IN MUMBAI")
        st_folium(map,width=1000)
    
    elif choice=="DASHBOARD":

        # Load sample data (you can replace this with your own dataset)
        data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\wastewise\Data.csv')

        # Title
        st.title('Dashboard')

        # Bar chart for visits
        st.bar_chart(data['States/UTs'])

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.scatter(data['States/UTs'], data['2016-17'], label='Visits')
            plt.xticks(rotation=90)  
            custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1
            plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
            plt.xlabel('States/UTs')
            plt.ylabel('2016-17')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.scatter(data['States/UTs'], data['2017-18'], label='Visits')
            plt.xticks(rotation=90)  
            custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1
            plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
            plt.xlabel('States/UTs')
            plt.ylabel('2017-18')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.scatter(data['States/UTs'], data['2018-19'], label='Visits')
            plt.xticks(rotation=90)  
            custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1
            plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
            plt.xlabel('States/UTs')
            plt.ylabel('2018-19')
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.scatter(data['States/UTs'], data['2019-20'], label='Visits')
            plt.xticks(rotation=90)  
            custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1
            plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
            plt.xlabel('States/UTs')
            plt.ylabel('2019-20')
            st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(data['States/UTs'], data['2020-21'], label='Visits')
        plt.xticks(rotation=90)  
        custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1   plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
        plt.xlabel('States/UTs')
        plt.ylabel('2020-21')
        st.pyplot(fig)
            

        # Line chart for visits by week of year
        plt.figure(figsize=(10, 4))
        plt.plot(data['States/UTs'], data['2016-17'], label='Visits')
        plt.xticks(rotation=90)  
        custom_x_positions = range(len(data['States/UTs']))  # Creates positions from 0 to length-1
        plt.xticks(custom_x_positions, data['States/UTs'])  # Set positions and labels
        plt.xlabel('States/UTs')
        plt.ylabel('Average Plastic Waste per State')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)



        traffic_sources = pd.DataFrame({
            'Source': ['Maharashtra','Tamil Nadu','Gujarat','Uttar Pradesh','West Bengal'
        ],
            'Percentage': [15, 12,11, 8, 4]
        })

        col1, col2 = st.columns(2)

        with col1:
            fig_traffic_sources = plt.figure(figsize=(3, 3))
            plt.pie(traffic_sources['Percentage'], labels=traffic_sources['Source'], autopct='%1.1f%%')
            plt.title("Top 5 Plastic Waste Producers in India")
            st.pyplot(fig_traffic_sources)

        with col2:
            img=Image.open(r'C:\Users\admin\OneDrive\Desktop\wastewise\Heatmap.png')
            new_size=img.resize((700,500))
            st.image(new_size)

    elif choice=="INFORMATION":
        video_file = open(r'C:\Users\admin\OneDrive\Desktop\wastewise\info.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

