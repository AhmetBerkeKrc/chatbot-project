import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# Setup Google Sheets API access with credentials
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("appointment_credential.json", scopes=scopes)
client = gspread.authorize(creds)

# Open the spreadsheet and select the worksheet
sheet_id = "SHEET_ID"
sheet = client.open_by_key(sheet_id)
worksheet = sheet.worksheet("app_sheet")

# Load data from the sheet into a pandas DataFrame
def load_data():
    values = worksheet.get_all_values()
    columns = values[0]
    return pd.DataFrame(values[1:], columns=columns)

# Display available hours from the DataFrame
def available_schedule(df):
    available_hours = df[df["Status"] == "Available"]
    print("\nAvailable Hours:")
    print(available_hours[["Doctor Name", "Date", "Time"]])
    return available_hours

# Handle the appointment booking process
def appointment(available_hours, df, worksheet):
    user_input = input("Provide doctor name and appointment time (Doctor Name, Time): ")
    try:
        doctor_name, appt_time = user_input.split(", ")
        
        if doctor_name in available_hours["Doctor Name"].values and appt_time in available_hours[available_hours["Doctor Name"] == doctor_name]["Time"].values:
            user_info = input("Provide your info (Full Name, ID Number, Phone Number): ")
            user_name, user_id, user_phone = user_info.split(", ")
            print(f"Dear {user_name}, your appointment from Dr.{doctor_name} at {appt_time} has been reserved.")
            update_df(doctor_name, appt_time, user_name, user_id, user_phone, df, worksheet)
        else:
            print("Invalid doctor name or time.")
    except IndexError:
        print("Invalid input format. Please try again.")

# Update the DataFrame and Google Sheet with the booked appointment
def update_df(doctor_name, appt_time, user_name, user_id, user_phone, df, worksheet):
    mask = (df["Doctor Name"] == doctor_name) & (df["Time"] == appt_time)
    df.loc[mask, "Patient Full Name"] = user_name
    df.loc[mask, "Patient ID Number"] = user_id
    df.loc[mask, "Patient Phone Number"] = user_phone
    df.loc[mask, "Status"] = "Booked"

    updated_values = [df.columns.tolist()] + df.values.tolist()
    worksheet.clear()
    worksheet.update("A1", updated_values)

# Main loop to continuously load data and process user input
while True:
    df = load_data()
    available_hours = available_schedule(df)
    appointment(available_hours, df, worksheet)
