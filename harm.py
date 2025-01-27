import joblib
import streamlit as st
import time
        
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

def progress_bar():
	progress_text = 'Detecting harm.'
	my_bar = st.progress(0, text=progress_text)

	for percent_complete in range(100):
    		time.sleep(0.01)
    		my_bar.progress(percent_complete + 1, text=progress_text)
	time.sleep(1)
	my_bar.empty()

def test_comment(comment):
    comment_vector = vectorizer.transform([comment])
    prediction = model.predict_proba(comment_vector)
    harmful_prob = prediction[0][0]  # Probability of harmful class
    
    if harmful_prob > 0.5: 
        st.empty()
        st.write(f'Harmful comment detected. (Probability: {harmful_prob:.2%})')
    else:
        st.empty()
        st.write(f'Comment is not harmful. (Probability: {harmful_prob:.2%})')
        
new_comment = st.text_area('Comment something to test harm speech.')

if st.button('Is this hate?'):
	progress_bar()
	test_comment(new_comment)
