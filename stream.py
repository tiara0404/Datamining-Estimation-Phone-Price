import pickle
import streamlit as st 

# membaca model
phone_model = pickle.load(open('estimasi_harga_handphone.sav','rb'))

#judul web
st.title('Aplikasi Prediksi Harga Handphone')


Storage = st.number_input('Input Jumlah Storage/Penyimpanan HP :')
BatteryCapacity  = st.number_input('Input Kapasitas BatteryCapacity :')
ScreenSize  = st.number_input('Input ScreenSize :')
RAM = st.number_input('Input Ukuran RAM :')
JumlahKamera = st.number_input('Input Jumlah Kamera :')

#code untuk estimasi
phone_est=''

#membuat button
if st.button('Estimasi Harga'):
    phone_pred = phone_model.predict([[BatteryCapacity,ScreenSize,RAM,Storage,JumlahKamera]])

    st.success(f'Estimasi Harga : {phone_pred[0]:.2f}')