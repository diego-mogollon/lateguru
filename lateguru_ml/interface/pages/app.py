import streamlit as st
import datetime

origin_airports = ['LAX', 'ATL', 'DEN', 'DFW', 'ORD']
dest_airports = ['JFK', 'HNL', 'DFW', 'CLT', 'KOA', 'PHX', 'OGG', 'ORD', 'MIA',
       'PHL', 'LAS', 'DCA', 'AUS', 'MCO', 'BNA', 'BOS', 'SEA', 'FLL',
       'PDX', 'EWR', 'GEG', 'RSW', 'SFO', 'TPA', 'ANC', 'CHS', 'BUF',
       'RIC', 'RNO', 'SLC', 'BDL', 'MTJ', 'RDU', 'PBI', 'BZN', 'ATL',
       'LIH', 'DTW', 'MSP', 'DEN', 'CVG', 'MEM', 'MSY', 'PVU', 'BOI',
       'MFR', 'EUG', 'BLI', 'TUL', 'SGF', 'BWI', 'CLE', 'IAH', 'MCI',
       'PIT', 'OAK', 'EGE', 'OKC', 'JAC', 'ASE', 'SAN', 'PRC', 'SMF',
       'RDM', 'SBP', 'HDN', 'PSP', 'ACV', 'FAT', 'MRY', 'SUN', 'DAL',
       'SJC', 'STS', 'ABQ', 'TUS', 'SAT', 'MSO', 'IAD', 'HOU', 'MDW',
       'STL', 'ELP', 'FCA', 'OMA', 'SBA', 'COS', 'RDD', 'ITO', 'SBN',
       'IND', 'SHV', 'GRR', 'LGA', 'SDF', 'MKE', 'CMH', 'JAX', 'DSM',
       'XNA', 'BIH', 'PAE', 'GJT', 'BIL', 'DRO', 'SAF', 'ICT', 'LIT',
       'FAR', 'MFE', 'CID', 'RAP', 'IDA', 'PSC', 'FSD', 'SCK', 'BTR',
       'MSN', 'SNA', 'ORF', 'GUC', 'ONT', 'LAX', 'BUR', 'SJU', 'EYW',
       'STT', 'GRK', 'CRP', 'JAN', 'ACT', 'SAV', 'AEX', 'TYR', 'CAE',
       'PNS', 'GNV', 'BRO', 'BHM', 'SJT', 'BPT', 'HSV', 'LFT', 'COU',
       'LCH', 'LBB', 'MDT', 'MLU', 'CHA', 'SWO', 'SRQ', 'GSP', 'TXK',
       'VPS', 'CMI', 'TYS', 'AMA', 'MOB', 'GGG', 'ABI', 'LRD', 'GRI',
       'ILM', 'LEX', 'LAW', 'SPS', 'MYR', 'MHK', 'DAY', 'EVV', 'CLL',
       'SPI', 'MGM', 'AGS', 'GCK', 'FSM', 'GPT', 'DRT', 'DAB', 'ECP',
       'BIS', 'HRL', 'JLN', 'TRI', 'FLG', 'FWA', 'BMI', 'TLH', 'YUM',
       'ROW', 'GSO', 'BFL', 'MLI', 'PIA', 'MAF', 'AVL', 'SGU', 'BTV',
       'LGB', 'TVC', 'SYR', 'FAY', 'CSG', 'PWM', 'FAI', 'BGR', 'MLB',
       'HHH', 'GRB', 'ALB', 'ACY', 'ROC', 'CPR', 'DIK', 'MOT', 'GTF',
       'HLN', 'LNK', 'LBL', 'RST', 'BFF', 'CNY', 'GCC', 'PUB', 'LBF',
       'EAR', 'ALS', 'HYS', 'LAR', 'SLN', 'VEL', 'RKS', 'JMS', 'ATY',
       'SUX', 'SHR', 'RIW', 'DDC', 'XWA', 'PIR', 'CYS', 'COD', 'ATW',
       'TWF', 'LWS', 'FOD', 'OTH', 'HOB', 'BKG', 'WYS', 'BTM', 'DVL',
       'PVD', 'ABE', 'CHO', 'OAJ', 'DHN', 'VLD', 'BQK', 'ABY', 'ROA',
       'CRW', 'GTR', 'STX', 'TTN', 'HPN', 'ISP', 'SCE', 'SWF', 'ALO',
       'AZO', 'LSE', 'FNT', 'TOL', 'CWA', 'LAN', 'DBQ', 'AVP', 'MQT',
       'MHT', 'MKG', 'JST', 'PAH', 'CMX', 'CGI', 'EAU', 'OGS', 'SHD',
       'LWB', 'CKB', 'DEC', 'DLH', 'MCW', 'TBN', 'MBS', 'ACK', 'RHI',
       'CAK', 'MVY', 'ERI']
carriers = ['American Airlines Inc.', 'Alaska Airlines Inc.',
       'JetBlue Airways', 'Delta Air Lines Inc.',
       'Frontier Airlines Inc.', 'Allegiant Air',
       'Hawaiian Airlines Inc.', 'Spirit Air Lines',
       'SkyWest Airlines Inc.', 'Horizon Air', 'United Air Lines Inc.',
       'Southwest Airlines Co.', 'Endeavor Air Inc.', 'Envoy Air',
       'PSA Airlines Inc.', 'Mesa Airlines Inc.', 'Republic Airline']



st.write("""# :zap: :cloud: **Welcome to Lateguru** :airplane:""")


st.write('''### *Predict flight delays so that you can navigate your way to the departure gate without being in a rush*''')

st.write('###### Enter in the below details to predict whether your flight is likely to be delayed:')

origin_picker = st.selectbox('Enter in your origin airport', origin_airports)
dest_picker = st.selectbox('Enter in your destination airport', dest_airports)
carrier_picker = st.selectbox('Enter in your airline carrier', carriers)
date_picker = st.date_input("What is your date of departure?")
time_picker = st.time_input("What is your scheduled flight departure time?")

if st.button('Predict whether your flight will be delayed'):
    print('button clicked!')
    st.write('Your flight is:')

    #code in logic so that if the prediction result from the model is 1,
    #then the flight is delayed, otherwise 0 == value of 'not delayed'