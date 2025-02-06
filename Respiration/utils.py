import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    return

def extract_patientID(patient_path):
    return patient_path.split("/")[5].split("_")[0]