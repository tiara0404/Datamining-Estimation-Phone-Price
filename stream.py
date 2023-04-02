import pickle
import streamlit as st 
import setuptools
# membaca model
phone_model = pickle.load(open('estimasi_harga_hp.sav','rb'))

#judul web
st.title('Aplikasi Prediksi Harga Handphone')

col1, col2,col3=st.columns(3)
with col1:
    Storage = st.number_input('Input Jumlah Storage :')
with col2:
    BatteryCapacity  = st.number_input('Input Kapasitas BatteryCapacity :')
with col3:
    ScreenSize  = st.number_input('Input ScreenSize :')
with col1:
    RAM = st.number_input('Input Ukuran RAM :')
with col2:
    JumlahKamera = st.number_input('Input Jumlah Kamera :')
with col3:
    cam1 = st.number_input('Input Resolusi Kamera Pertama :')
with col1:
    cam2 = st.number_input('Input Resolusi Kamera Kedua :')
with col2:
    cam3 = st.number_input('Input Resolusi Kamera Ketiga :')
with col3:
    cam4 = st.number_input('Input Resolusi Kamera Keempat :')
with col1:
    idBrand = st.number_input('Input id Brand Handphone')
with col2:
    st.caption('''
        id Brand : \n
        CAT = 1 \n
        Sony = 2 \n
        Blackberry = 3 \n
        LG = 4 \n
        Asus = 5 \n
        Google = 6 \n
        Huawei = 7 \n
        Oneplus = 8 \n
        ''')
with col3:
    st.caption('''
        id Brand : \n
        Motorolla = 9 \n
        Nokia = 10 \n
        Apple = 11 \n
        Vivo = 12 \n
        Realme = 13 \n
        Oppo = 14 \n
        Xiaomi = 15 \n
        Samsung = 16''')
#code untuk estimasi
phone_est=''

#membuat button
with col1:
    if st.button('Estimasi Harga'):
        phone_pred = phone_model.predict([[BatteryCapacity,ScreenSize,RAM,Storage,JumlahKamera,cam1,cam2,cam3,cam4,idBrand]])

        st.success(f'Estimasi Harga : {phone_pred[0]:.2f}')