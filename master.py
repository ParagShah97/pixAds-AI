import streamlit as st
import pandas as pd
from PIL import Image
import os
import overall_pred as op
import advertisments as ads
import random
from collections import Counter
from time import sleep

# Define the initialization function
def initialize():
    if 'logged_in' not in st.session_state:
        # Session state to check if the user is logged in or not.
        st.session_state['logged_in'] = False
        # Session state to check if the applicaton need to re run again or not.
        st.session_state['app_already_run'] = False
        # Session state for storing the previously stored label.
        st.session_state['predicted_label_prev'] = ''
        st.session_state['models_instance'] = op.LabelPred()
        

# Call the initialization function at the beginning of the app start.
initialize()

# Function to handle login
def login(username, password):
    if username == 'paragsha' and password == 'paragsha':
        st.session_state['logged_in'] = True
       
# delete_image: method removes the images from the folder location.
def delete_image(image_path):
    print('image path', image_path)
    try:
        os.remove(image_path)  # Delete the image file
        # st.success(f"Image '{file}' deleted successfully.")
        st.session_state['app_already_run'] = False
        st.session_state['predicted_label_prev'] = ''
        st.experimental_rerun()
    except PermissionError:
        st.error("Error: You don't have permission to delete the image.")

# adv_pred: Method will predict the advertisment for the uploaded images.
def adv_pred(folder_path):
    label = []
    # label = ''
    with st.spinner("Processing"):
        for root, _, files in os.walk(folder_path):
            if len(files) == 0 or st.session_state['app_already_run']:
                print('loop breaker run')
                break
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for common image extensions
                    LabelPredInst = op.LabelPred()
                    # st.session_state['models_instance'] = LabelPredInst
                    image_path = os.path.join(root, file)
                    # label = LabelPredInst.predict(image_path)
                    label.append(LabelPredInst.predict(image_path))
    
    if len(label) != 0:
        word_counts = Counter(label)
        print('returned word count from master', word_counts)
        most_frequent_word = word_counts.most_common(1)[0][0]
        return most_frequent_word
    else:
        return ''
    # return label
    
# save_uploaded_image: method will save the uploaded file to a folder.
def save_uploaded_image(uploaded_file, save_folder="images"):
  if uploaded_file is not None:
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)  # Handles existing folder gracefully

    # Generate a unique filename (timestamp + original filename)
    filename = f"filename_{uploaded_file.name}"
    save_path = os.path.join(save_folder, filename)

    try:
      with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
      return f"Image '{filename}' saved successfully!"
    except Exception as e:
      return f"Error saving image: {str(e)}"
  else:
    return "Please upload an image file."

# Main application
# Condition (track) for logged in user.
if st.session_state['logged_in']:
    tab1, tab2 = st.tabs(["Home", "About Application"])
    
    imges = []
    with tab1:
        st.title('Piks~AI')
        # st.write(f"#### Let us know how's your day, we will perk you with the best you can get!")
        st.write(f"#### Let us know how your day is going, and we'll brighten it with the best personalized offers just for you!")
        with st.expander("Discover Personalized Ads with Every Upload"):
            st.write(f"### Intoduction to Piks!")
            st.markdown(f'<div font_size: 14px; style="text-align: justify; margin-bottom: 8px"> {ads.introduction_to_application} </div>  ', unsafe_allow_html=True)
            
        # st.button('Refresh')
        with st.form("/upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Post what's on your mind!", type=["jpg", "png"])
            file_submitted = st.form_submit_button("Upload the selected file!")
            if file_submitted and (uploaded_file is not None):
                message = save_uploaded_image(uploaded_file)
                st.write(message)
                st.session_state['app_already_run'] = False
        
        
        folder_path = 'images'
        label_to_predict = ''
        current_label = adv_pred(folder_path)
        label_to_predict = current_label if len(current_label) > 0 else st.session_state['predicted_label_prev']
        # print('******', label_to_predict)
        # Walk over the folder location to render the images over the screen.
        try:
            for root, _, files in os.walk(folder_path):
                i = 0
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for common image extensions
                        image_path = os.path.join(root, file)
                        with Image.open(image_path) as image:
                            # Display image with filename as caption and optional width
                            # _, col2_pic, _ = st.columns([1, 1, 1])
                            # with col2_pic:
                            st.image(image, caption=os.path.splitext(file)[0], width=600)

                            # Create like and delete buttons with unique keys (using f-strings)
                            like_button_name = f"like_button_{i}"
                            delete_button_name = f"delete_button_{i}"
                            confirm_button_name = f"confirm_button_{i}"
                            # Columns for image action button.
                            col_img1, _, col_img2 = st.columns([1, 1, 1])
                            if col_img1 is not None:
                                with col_img1:
                                    if st.button("Like", key=like_button_name):
                                        # Handle like button click logic here
                                        pass
                            if col_img2 is not None:
                                with col_img2:
                                    if st.button("Delete", key=delete_button_name):
                                        uploaded_file = None
                                        delete_image(image_path)                                       
                            st.empty()
                            i += 1

        except FileNotFoundError:
            st.error(f"Error: Folder '{folder_path}' not found.")
        
        if label_to_predict != None:
            print('===>>>', label_to_predict)
            # label_to_predict is non empty then only show the ads section.
            if label_to_predict:
                st.session_state['predicted_label_prev'] = label_to_predict
                # Select the random ads from the ads pool.
                random_video = random.choice(ads.urls_dict[label_to_predict]['videos'])
                random_images = random.choice(ads.urls_dict[label_to_predict]['images'])
                # print('=====', random_video, random_images)
                st.markdown("""<hr style="border-top: 1px solid #ddd;"/>""", unsafe_allow_html=True)
                st.markdown(f'<div font_size: 14px; style="text-align: center; margin-bottom: 8px"> ♡ {ads.predicted_label_messages[label_to_predict]} ♡ </div>  ', unsafe_allow_html=True)
                st.markdown("""<hr style="border-top: 1px solid #ddd;"/>""", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    video_path = random_video
                    st.video(video_path, loop=True)
                    st.empty()
                with col2:
                    st.image(random_images)      
                    st.empty()
            st.session_state['app_already_run'] = True
        
        st.markdown("""<hr style="border-top: 1px solid #ddd;"/>""", unsafe_allow_html=True)
        st.markdown("Made with :heart: by")
        col1_foot, col2_foot, col3_foot = st.columns(3)
        with col1_foot:
            st.markdown('Parag Shah')
            st.markdown('paragsha@buffalo.edu')
        with col2_foot:
            st.markdown('Sai Venkat Reddy')
            st.markdown('Ssheri@ubuffalo.edu')
        with col3_foot:
            st.markdown('Prajakta Jhade')
            st.markdown('pjhade@buffalo.edu')
        
    with tab2:
        with st.expander('About our Dataset', expanded=True):
            st.write(f"### Glimpse about MIT places dataset:")
            st.markdown(f'<div font_size: 14px; style="text-align: justify; margin-bottom: 8px"> {ads.about_dataset} </div>  ', unsafe_allow_html=True)
            # st.write()
            st.empty()
            st.write(f"### Class Information Table:")
            st.table(pd.DataFrame(ads.data))
            st.empty()
            resnet_model, vgg_model, densenet_model, alexnet_model = st.session_state['models_instance'].getModelInstance()

        
        with st.expander('Resnet Model Architecture'):
            st.write(resnet_model)
        with st.expander('VGG13 Model Architecture'):
            st.write(vgg_model)
        with st.expander('DenseNet Model Architecture'):
            st.write(densenet_model)
        with st.expander('AlexNet Model Architecture'):
            st.write(alexnet_model)
else:
    st.title("Piks~AI Login")
    
    with st.form(key='login_form'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        login_button = st.form_submit_button(label='Login')
    
    if login_button:
        login(username, password)
        if st.session_state['logged_in']:
            pass
        else:
            st.error('Login failed. Please try again.')






    




