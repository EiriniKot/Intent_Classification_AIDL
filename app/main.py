import streamlit as st
import requests
import random

endpoint = "http://localhost:8080/predictions/roberta_intent"

general_link = f'https://aidl.uniwa.gr/'

json_answers = {"Applications": [('I see, I think what you are looking for is in Applications Section', f'{general_link}applications/'),
                                 ('I believe you want information about application', f'{general_link}applications/')],
                "Contact": [('If you want to find contact information I suggest you to visit this', f'{general_link}contact/'),
                            ('Are you looking for some contact information?', f'{general_link}contact/'),
                            ('Are you looking for some contact information?', f'{general_link}contact/')],
                "Curriculum": [('You can find curriculum information under this link', f'{general_link}all-courses/'),
                                ('Information about Curriculum can be found here', f'{general_link}all-courses/')],
                "Documents": [('This page contains useful documents', f'{general_link}mscdocuments/'),
                              ('Are you looking for some document? Look here', f'{general_link}mscdocuments/')],
                "Fees": [('You should consider looking into the fees section here', f'{general_link}tuition-fees/'),
                          ('In this page you may find all the important info about the cost of the program', f'{general_link}tuition-fees/')],
                "Instructors": [('For more information about the instructors please refer here',  f'{general_link}instructors/'),
                                ('I see you are asking me a question about instructors here you can find all about them', f'{general_link}instructors/')],
                "Invited": [('I believe you are asking me about the invited lecturers here you can find some info',f'{general_link}invited-lecturers-2/'),
                            ('Here you go! This page is about the invided lecturers of the program', f'{general_link}invited-lecturers-2/')],
                "Irrelevant": [('I am sorry, I do not understand what you want, here is the website of the master!', general_link)],
                "Registry": [('Here you can enter the student registry', f'{general_link}student-registry/'),
                             ('I think you want to enter the student registry', f'{general_link}student-registry/')],
                "Schedule": [('I guess you are looking for the schedule so there you go!', f'{general_link}course-schedule/'),
                             'For the schedule you can search here', f'{general_link}course-schedule/']}

def get_prediction(input_data):
    """get predictions from api, return a response"""
    response = requests.post(endpoint, data={'body': input_data})
    response_b = response.content.decode()
    return response_b


if __name__ == "__main__":
    st.set_page_config(page_title="Chatbot", page_icon=":robot_face:", layout="wide")

    st.image('https://aidl.uniwa.gr/wp-content/uploads/2021/03/cropped-newLogo_1-180x45_trans_ar.png', width=300)
    st.title('Welcome to student information chatbox!')

    user_input = st.text_input("Enter your message:")
    if st.button('Send'):
        # Get the prediction results
        prediction_results = get_prediction(user_input)
        category = json_answers[prediction_results]
        get_answer = random.choice(category)
        answer = f'{get_answer[0]} [link]({get_answer[1]})'
    else:
        answer = ''

    st.write('<p style="font-size:20px; color:white;">Prediction Results:</p>',
             unsafe_allow_html=True)
    st.write(answer)

    st.markdown("""
            <style>
                .stMarkdown h1, h2, h3, h4, h5, h6 {
                    text-align: left;
                    color: #fff;
                    font-size: 23px;
                }
                .stMarkdown {
                    background-color: rgba(160,240,250,0.7);
                    padding: 13px;
                    border-radius: 13px;
                    margin-bottom: 13px;
                }
                .title {
                text-align: center;
                }
            </style>
            """, unsafe_allow_html=True)
