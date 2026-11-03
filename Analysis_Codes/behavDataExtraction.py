#### ====== BEHAV DATA EXTRACTION ===== ####

"""
This script allows for simple Data extraction. It extracts both behavioral and occular data from the ASC files
 and saves them in a convenient format for further analysis. 

Processing Pipeline:
1. 

"""


# This is a simple script to get from our ASC files behavioral Triggers for every trial and construct - trial-wise matrix of behavioral data from ALL subjects in DATA

import os
import pickle
from analysisFunctions import *
from matplotlib.backends.backend_pdf import PdfPages

workingDir = os.getcwd();

### Create Paths to all subfolders in which the data will be stored 
analysisPath = os.path.join(workingDir,'analysis',"Preprocessed");
rawDataPath  = os.path.join(workingDir,'analysis',"RawData");

dataPath = os.path.join(workingDir,'data')
dataQualityPath = os.path.join(workingDir,'analysis','dataQualityChecks')
if not os.path.isdir(dataPath): os.makedirs(dataPath)
if not os.path.isdir(analysisPath) : os.makedirs(analysisPath)
if not os.path.isdir(dataQualityPath): os.makedirs(dataQualityPath)

pdf_path_raweyegaze = os.path.join(dataQualityPath, "RawEyeGaze.pdf")
pdf_raweyegaze = PdfPages(pdf_path_raweyegaze)
pdf_path_raweyekde = os.path.join(dataQualityPath, "RawEyeKDE.pdf")
pdf_raweyekde = PdfPages(pdf_path_raweyekde)
pdf_path_rawpupil = os.path.join(dataQualityPath, "RawPupil.pdf")
pdf_rawpupil  = PdfPages(pdf_path_rawpupil)
pdf_path_gazecheck = os.path.join(dataQualityPath, "gazeCheckPars.pdf")
pdf_gazecheck  = PdfPages(pdf_path_gazecheck)



print("Working Directory: ", workingDir)
print("Current Directory: ", os.path.dirname(os.path.abspath(__file__)))


### Parameters of Analysis: 

# General Parameters

# -- Saving Parameters
paramsDict = {};

begID = 0
endID = len([f for f in os.listdir(dataPath) if  not os.path.isfile(os.path.join(dataPath, f))]) # Length of all the subjects, you can choose them! 



paramsDict = {}

### PREPROCESSING STAGES:
paramsDict['downs'    ] = 1   # 1) Downsampling
paramsDict['centering'] = 0   # 2) Reduce data to only Center (OFF)
paramsDict['binterp'  ] = 1   # 3) Blink Interpolation
paramsDict['outrmv'   ] = 1   # 4) Remove Outliers
paramsDict['nanrmv'   ] = 1   # 5) Remove Nans (zero pupil)
paramsDict['smth'     ] = 1   # 6) Smooth out

# General parameters
paramsDict['verbose'      ] = 1        # MUSI być 1 – inaczej downsampling się nie wykona (wcięcie w preprocessingPipeline)

### 1) Downsampling
paramsDict['resRate'      ] = 100      # Hz

### 2) Centering
paramsDict['dgvCent'      ] = 5        # deg (nieużywane, bo centering = 0)

### 3) Blink Interpolation
paramsDict['blinkInt'     ] = 0        # 0 = linear
paramsDict['maxblength'   ] = 500      # ms
paramsDict['blkboundry'   ] = 75       # ms
paramsDict['blkboundry2'   ] = 75       # ms

paramsDict['blinkbegbound'] = 1000     # ms

### 4) Smoothing
paramsDict['smoothwin'    ] = 10       # samples
paramsDict['maxcluslen'   ] = 2000     # ms
paramsDict['mingaplen'    ] = 50       # ms

### Pupil Diameter Computation
paramsDict['trialOffset'  ] = "all"
paramsDict['begTrialOff'  ] = 5000     # ms cut at the beginning of a trial
paramsDict['endTrialOff'  ] = 0        # ms cut at the end of a trial

### Baseline
paramsDict['baseOffset'   ] = 5000     # ms
paramsDict['baseType'     ] = "mean"

paramsDict['dataChecks'   ] = 1        # wykresy kontrolne (RawEyeKDE, RawGaze, RawPupil)
paramsDict['dataSaves'    ] = 0        # 0 = NIE nadpisuj RawData/<ID>/raw*.csv ani behavioralData.csv

### Save analysis Name with all the parameters
analysisName = f"Preprocessed_Data_2"

filehandler = open(os.path.join(analysisPath,analysisName,"analysisParameters.txt"), 'wt')
filehandler.write(str(paramsDict))
if os.path.isdir(os.path.join(analysisPath,analysisName)):
    print("Analysis Folder already exists! \n")
    input("Press Enter to continue and overwrite the data in this folder, or Ctrl+C to cancel the analysis")
else:
    os.makedirs(os.path.join(analysisPath,analysisName))


# Get Subjects: 
subjects = [f for f in os.listdir(dataPath) if  not os.path.isfile(os.path.join(dataPath, f))] # get all the filenames
subjects = subjects[begID:endID] # Fine-tune if needed to get specific Number of subjects: 

### Individual Data path for current analysis
individualPath =  os.path.join(analysisPath,analysisName,'individual')



allBehav = []
allEyeRaw = []
allBlinks  = []
allSaccades = []
allFixations = []
allEyePreproc = []
for i, subjName in enumerate(subjects):

    #==========================================#
    #### -->   PREPARE FOLDERS AND VARS   <--- ####
    #==========================================#

    ### load subject Data:
    subjPath = os.path.join(individualPath,subjName)
    os.makedirs(subjPath,exist_ok=True)
    os.makedirs(os.path.join(rawDataPath,subjName),exist_ok=True)

     # Define PDF paths
    pdf_path_downsampled = os.path.join(subjPath, "Downsampled_Pupil_Data.pdf")
    pdf_path_screen_only = os.path.join(subjPath, "Screen_Only.pdf")
    pdf_path_blink_rejection = os.path.join(subjPath, "After_Pupil_Artefacts_Rejection.pdf")
    pdf_path_outlier_rejection = os.path.join(subjPath, "After_Outlier_Rejection.pdf")
    pdf_path_centered = os.path.join(subjPath, "SmoothedPupil.pdf")
    pdf_path_nan_rejection = os.path.join(subjPath,"NaN_rejected_Pupil.pdf")
    # Open PdfPages objects for each PDF

    pdf_downsampled = PdfPages(pdf_path_downsampled)
    pdf_center_only = PdfPages(pdf_path_screen_only)
    pdf_blink_rejection = PdfPages(pdf_path_blink_rejection)
    pdf_outlier_rejection = PdfPages(pdf_path_outlier_rejection)
    pdf_nan_rejection = PdfPages(pdf_path_nan_rejection)
    pdf_smoothed = PdfPages(pdf_path_centered)


    # Create Subject Specific Folders and write initial Preambule there
    log_file = open(os.path.join(individualPath,subjName,subjName+"_preprocLog.txt"),'w+')
    log_file.write(" "*15+"="*58 + "\n")
    log_file.write(" "*15+"="*5+ " "*15+" PREPROCESSING LOG FILE"+ " "*15+"="*5+ " \n")
    log_file.write(" "*15+"="*58 + "\n\n")

    log_file.write("\n"+"-"*100 + "\n")
    log_file.write("PROCESSING SUBJECT: " + subjName + "  ======"+ "\n")
    log_file.write("\n"+"-"*100 + "\n")
    print("#-------- Current Subject: ", subjName, " --------#")
    log_file.write("      Participant ID: " + subjName + "\n")
    log_file.write("      Data Checks:    " + str(paramsDict['dataChecks']) + "\n")
    log_file.write("      Data Saves:     " + str(paramsDict['dataSaves']) + "\n")

    log_file.write("\n===# Load and Prepare Data #=== \n\n")

    

    #==========================================#
    #### -->   DATA EXTRACTION   <--- ####
    #==========================================#

    ### Read Behavioral Data
    log_file.write("    Loading Participant Data... ")
    [own,rawEye, events, gazeCoords] = loadParticipant(dataPath, name=subjName) # Gets my RawEye Data as well as events and gaze Coordination
    RawDF = pd.DataFrame(rawEye,columns=["TimePoint",'LeftX','LeftY','LeftPupil','RightX','RightY','RightPupil']);
    log_file.write("done \n")

    # Get Occular Events
    log_file.write("    Getting Events...           ")
    fixationDF,saccadesDF,blinkDF = getEvent(events)
    log_file.write("done \n")

    # Change Timing so it will be relative from the start of the experiment
    log_file.write("    Changing Timing....         ")
    RawDF,fixationDF,saccadesDF,blinkDF,own = changeTiming(RawDF,fixationDF,saccadesDF,blinkDF,own)
    log_file.write("done \n")


    # Parse your own events getting a Wide format getting only PARTS with their Timing  (Maybe Write Checks?)
    log_file.write("    Getting Own Events...       ")
    event_dfs = parse_events(own)
    event_dfs = rename_parts(event_dfs) # 3Be sure that the numeration for each entity part is Correct! 
    log_file.write("done \n")
    if len(event_dfs) != 40:
        log_file.write("    WARNING! the events are not whole!       ")
    log_file.write("\n            Events DataFrame:       \n")
    log_file.write(event_dfs.describe().to_string())


    # Contains All our Data but now cONVIENIENT Split for a stories, so Only parts that were listened to are considered
    log_file.write("\n\n    Splitting by events...      ")
    split_data = split_dataframes_by_events(RawDF, fixationDF, saccadesDF, blinkDF, event_dfs)
    log_file.write("done \n")

    ### Save and preprocess Behavioral Data: 
    event_dfs['Length'] = event_dfs['end'] - event_dfs['beg'] #Length as a difference between End and Beginning
    event_dfs['PressTime'] = event_dfs['preftime']  - event_dfs['end'] #Press Time as a difference between End and Beginning
    event_dfs['Pref_relative'] = (event_dfs['preference'] - np.mean(event_dfs['preference']))/np.std(event_dfs['preference']) # Z-scored Preference
    event_dfs['MW_relative'] =  (event_dfs['key'] - np.mean(event_dfs['key']))/np.std(event_dfs['key']) # Z-scored MW
    event_dfs['MW'] = event_dfs['key'] # Non Relative MW
    event_dfs['trialNum'] = range(0,40) # Trial Number
    event_dfs['Hero'] = [event_dfs['Part'][i].split('_')[0] for i in range(0,40)] #About Which Hero is the current trial about
    event_dfs['Subject'] = subjName # Subject Name (fro later Joining Purposes)
    event_dfs['DS'] = f"DS{int(i+1):02d}" # DS for better joining of all the dataframes later on
    event_dfs['fragment'] = event_dfs['Part'] # Name of the fragment 
    currentBehavDf = event_dfs[['DS','trialNum','Subject','fragment','Hero','Length','MW','MW_relative','preference','Pref_relative','recallfirst']]
    currentBehavDf.to_csv(os.path.join(subjPath,"currentBehav.csv"))
    allBehav.append(currentBehavDf)

    ### Save and Extract raw Eye  Data
    rawDf = extractOccularData(split_data['STORY_1']['RawDF'],subjName)
    allEyeRaw.append(rawDf)

    ### Save and Extract Saccade  Data
    saccadesDF = extractOccularData(split_data['STORY_1']['saccadesDF'],subjName)
    allSaccades.append(saccadesDF)

    ### Save and Extract Blink    Data
    blinkDF = extractOccularData(split_data['STORY_1']['blinkDF'],subjName)
    allBlinks.append(blinkDF)

    ### Save and Extract Fixation Data
    fixationDF = extractOccularData(split_data['STORY_1']['fixationDF'],subjName)
    allFixations.append(fixationDF)

    if paramsDict['dataSaves']:
        fixationDF.to_csv(os.path.join(os.path.join(rawDataPath,subjName),"rawFixations.csv"))
        blinkDF.to_csv(os.path.join(os.path.join(rawDataPath,subjName),"rawBlinks.csv"))
        rawDf.to_csv(os.path.join(os.path.join(rawDataPath,subjName),"rawEye.csv"))
        saccadesDF.to_csv(os.path.join(os.path.join(rawDataPath,subjName),"rawSaccades.csv"))
        currentBehavDf.to_csv(os.path.join(os.path.join(rawDataPath,subjName),"behavioralData.csv"))



    #==========================================#
    #### -->   GENERAL DATA OVERVIEW   <--- ####
    #==========================================#

    log_file.write("\n===# Stats and Checks for Experiment Relevant only Data #=== \n\n ")

    [RawDFPupilTrue,_] = GeneralChecks(rawDf, blinkDF, saccadesDF, fixationDF, gazeCoords,log_file)
    
    ### Create List of Trial-wise DFs for Blinks, Saccades, Fixations and Raw Data:
    blinkDFs = list(split_data['STORY_1']['blinkDF'].values())
    RawDFs = list(split_data['STORY_1']['RawDF'].values())
    saccadesDFs = list(split_data['STORY_1']['saccadesDF'].values())
    fixationDFs = list(split_data['STORY_1']['fixationDF'].values())

    ### Check Trial-wise metrics
    gazeDiff, pupilDiff, saccDiff, blinkDiff, eyeposDiff, nogazeRate,pupil_mean,mean_eye_diff = qualiryMetrics(RawDFs, saccadesDFs, blinkDFs, gazeCoords, log_file=log_file,verbose=1)

    ### Make a Trial-wise metrics into a Descriptive Quality File. 
    quality_checks_to_excel( subjName, event_dfs,RawDFs,  saccadesDFs,  blinkDFs,  gazeCoords, dataQualityPath, excel_name="Descriptives_all.xlsx",   trial_col="trialNum")
    

    ### Plot Data if you'd like 
    if paramsDict['dataChecks'] == 1:
        print("     Plotting General Eye Behaviour during whole experiment for this subject")
        fig1 = eyeRawKDE(rawDf,titl=subjName)
        pdf_raweyekde.savefig(fig1)
        plt.close(fig1)

        fig2 =plotRawGaze(rawDf,gazeCoords,titl=subjName)
        pdf_raweyegaze.savefig(fig2)
        plt.close(fig2)

        fig3 =plotPupilTimecourse(rawDf, f"{subjName}: Raw Pupil Data with Blinks", blinkDF, saccadesDF, fixationDF, chooseViz='001')
        pdf_rawpupil.savefig(fig3)
        plt.close(fig3)


    #==========================================#
    #### -->   PUPIL DATA EXTRACTION   <--- ####
    #==========================================#

    log_file.write("\n===# Preprocessing the Data #=== \n\n ")
    log_file.write(f"    Preproc Parameters: General: \n")
    log_file.write(f"        Resampling Rate:      {paramsDict['resRate']}\n")
    log_file.write(f"        DGV from Center:      {paramsDict['dgvCent']}\n\n")
    log_file.write(f"    Preproc Parameters: Blinks: \n")
    log_file.write(f"        Blink Interpolation:  {paramsDict['blinkInt']}\n")
    log_file.write(f"        Maximum Blink Length: {paramsDict['maxblength']}\n")
    log_file.write(f"        Blink Interp Bound:   {paramsDict['blkboundry']}\n")
    log_file.write(f"        Blink Begining Bound: {paramsDict['blinkbegbound']}\n\n")
    log_file.write(f"    Preproc Parameters: Smooth: \n")
    log_file.write(f"        Window size:          {paramsDict['smoothwin']}\n")
    log_file.write(f"        Minimal Cluster len:  {paramsDict['maxcluslen']}\n")
    log_file.write(f"        Maximal Gap Length:   {paramsDict['mingaplen']}\n\n")


    ### Iterate Through Trials
    currentFinalData = []
    print("\n===# Preprocessing the Data #=== \n\n ")
    for i,(rawTrialDF,saccadeTrialDF,blinkTrialDF,fixationTrialDF) in enumerate(zip(RawDFs,saccadesDFs,blinkDFs,fixationDFs)):
        print(f"     Current Trial {i} ({currentBehavDf['fragment'].iloc[i]}) length: {len(rawTrialDF)*2} ms")
        finalData = preprocessingPipeline(blinkTrialDF,rawTrialDF,saccadeTrialDF,gazeCoords,"",currentBehavDf['fragment'].iloc[i],log_file=log_file,pdfs=[pdf_downsampled,pdf_center_only,pdf_blink_rejection ,pdf_outlier_rejection ,pdf_nan_rejection,pdf_smoothed],
                                            verbose   =paramsDict['verbose'],
                                            downs     =paramsDict['downs'    ] , resampleRate=paramsDict['resRate'],
                                            centering =paramsDict['centering'] , dgvCenter=paramsDict['dgvCent'],
                                            binterp   =paramsDict['binterp'  ] , interp_type=paramsDict['blinkInt'],interpBoundary=paramsDict['blkboundry'],interpBoundary2=paramsDict['blkboundry2'],maxBlinkDur=paramsDict['maxblength'],
                                            outrmv    =paramsDict['outrmv'   ] ,
                                            nanrmv    =paramsDict['nanrmv'   ] ,
                                            smth      =paramsDict['smth'     ] , smoothwin=paramsDict['smoothwin'], min_cluster_duration=paramsDict['maxcluslen'],max_gap_duration=paramsDict['mingaplen'])
            


        ### Baseline Correction and all other stuff should be computed in a different preprocessing Functions in order to reduce future computation time! 
        finalData['trialNum'] = i;
        finalData['fragment'] = currentBehavDf['fragment'].iloc[i]
        finalData['Subject'] = subjName
        finalData['DS'] = event_dfs['DS'].iloc[i]
        currentFinalData.append(finalData)

        ### Final Checks
        log_file.write("\n        Final Data Stats: \n")
        log_file.write(finalData.describe().to_string().replace('\n', '\n\t\t').join(['\t\t', ''])) # Adds two Tabulations!
        log_file.write("\n\n")
        log_file.write("        Sanity Checks on Final Data: \n")
        log_file.write(f"            Number of Non-missing Data events: {len(finalData['LeftPupil'].dropna())} ({len(finalData['LeftPupil'].dropna())/len(finalData)} %)\n")
        gazeD, _, _, _, _, _,pupil_mean,mean_eye_diff  = qualiryMetrics([finalData], [saccadesDF], [blinkDF], gazeCoords, log_file,title=f"         Quality Metrics: {currentBehavDf['fragment'].iloc[i]}")


    pdf_downsampled.close(); pdf_center_only.close(); pdf_blink_rejection.close(); pdf_outlier_rejection.close(); pdf_nan_rejection.close(); pdf_smoothed.close();


    subjWiseData = pd.concat(currentFinalData)
    allEyePreproc.append(subjWiseData)
    
    with open(os.path.join(subjPath,'Preprocessed.pkl'), 'wb') as handle:
        pickle.dump(subjWiseData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ### Save csv with SubjectWise behavioral and occular Data as well as fixation, saccade and blink Df in csv



with open(os.path.join(analysisPath,analysisName,'Preprocessed_All.pkl'), 'wb') as handle:
    pickle.dump(allEyePreproc, handle, protocol=pickle.HIGHEST_PROTOCOL)

pdf_raweyegaze.close()
pdf_rawpupil.close()
pdf_raweyekde.close()

###  Save concatenated Raw Data
if paramsDict['dataSaves']:
    pd.concat(allBehav,axis=1).to_csv(os.path.join(subjPath,"allBehav.csv"))

    with open(os.path.join(analysisPath,analysisName,'Preprocessed_All_bundle.pkl'), 'wb') as handle:
        pickle.dump(pd.concat(allBehav,axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pd.concat(allEyePreproc,axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pd.concat(allSaccades,axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pd.concat(allFixations,axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pd.concat(allBlinks,axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)

