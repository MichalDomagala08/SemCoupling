"""
This script preprocesses the transcribed audio files of the participant into a Metrics of the stories at hand.

It is then used to extract the data from the stories and save into excel files


The Analysis is computing

1. Story Descriptive Statistics
2. Segments Story into Lemmas
3. Computes Semantic Decoupling as a Story-Fragment Similarity on 2 occasions 
"""

import os,ast
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

analysisName = "Main_Analysis_1"
dataFolder = os.path.join(os.getcwd(),"analysis","RawData")

storyData =pd.read_excel(os.path.join(dataFolder,"recallAllData.xlsx"))
behavData = pd.read_excel(os.path.join(dataFolder,"behavAllData.xlsx"))

dfAnalysis = pd.merge(storyData,behavData, on=['Subject','fragment'],how='inner')

### (0) Story Quality Checks TBD
storyData['Lemmas']    = storyData['Lemmas'].apply(ast.literal_eval)
storyData['Sentences'] = storyData['Sentences'].apply(ast.literal_eval)
storyData['Verbs']     = storyData['Verbs'].apply(ast.literal_eval)
storyData['Nouns']     = storyData['Nouns'].apply(ast.literal_eval)

# Checking for the number of words Unknown by the Lemmatizer, and the number of words that are not in the dictionary.

### (1) Lexical Diversity Extraction
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
from storyTools import lexical_diversity, lexical_diversity_storywise

Ldiv = []; LdivS = []; LdivSS = []
allLemas = storyData['Lemmas'].sum()  # All Lemmas in the dataset
SLemmas = [ storyData[storyData['Subject'] == s]['Lemmas'].sum() for s in  storyData['Subject'].unique() ]                                      # All lemma list for a specific subject
for i,t in enumerate(storyData['Utterance']):
    Ldiv.append(lexical_diversity(storyData['Lemmas'].iloc[i]))                           # Lexical Diveristy inside an Utterance
    LdivSS.append(lexical_diversity_storywise(storyData['Lemmas'].iloc[i],allLemas))      # Lexical Diversity Based on Whole Lemmas
    LdivS.append(lexical_diversity_storywise(storyData['Lemmas'].iloc[i],SLemmas[list(storyData['Subject'].unique()).index(storyData['Subject'].iloc[i])]))      # Lexical Diversity based on Subject


dfAnalysis['LexDiv'] = Ldiv     # Lexical Div of a given fragment
dfAnalysis['LexDivS'] = LdivS   # Lexical Div of a fragment relative to lemmas ofa single subject
dfAnalysis['LexDivSS'] = LdivSS # Lexical Div of a Fragment relative to All Lemmas in the dataset

### (2) Autobiographical Featues 
from storyTools import autobiographyAnalysis
import morfeusz2
morf = morfeusz2.Morfeusz()
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
personalVerbsCount = [];
personalPronCount = [];
for i,t in enumerate(storyData['Utterance']):
    if type(t) is not str:
        personalVerbsCount.append(0)
        personalPronCount.append(0)
    else:
        personalVerbs, personalPron = autobiographyAnalysis(t,morf)
        personalVerbsCount.append(len(personalVerbs))
        personalPronCount.append(len(personalPron))

dfAnalysis['pVerbCount'] = personalVerbsCount     # Lexical Div of a given fragment
dfAnalysis['pPronCount'] = personalPronCount   # Lexical Div of a fragment relative to lemmas ofa single subject

### (3) Within Utterance Concept Embedding Extraction
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
# This metric is outdated - here for historical purposes. It is not used in the final analysis.
from gensim.models import fasttext as ftx
import fasttext
from sentence_transformers import SentenceTransformer,util
from storyTools import bertSentenceCosineSim, wmd_gensim
import torch

### Load Models
if "kv" not in locals():            kv          = ftx.load_facebook_vectors(r"C:\Users\barak\Documents\Python_Scripts\CoproraTools\word2vec\w2v_pl.bin");
if 'model_w2v'  not in locals():    model_w2v   = ftx.load_facebook_model(r"C:\Users\barak\Documents\Python_Scripts\CoproraTools\word2vec\w2v_pl.bin");
if 'model_sBERT'not in locals():    model_sBERT = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2');

bertSimilarity = []; fasttextSimilarity = []; meanSimSubject = []; meanSimDataset = []

for i,t in enumerate(storyData['Lemmas'].values):
    if type(t) is not str:
        bertSimilarity.append(np.nan);fasttextSimilarity.append(np.nan);meanSimSubject.append(np.nan);meanSimDataset.append(np.nan)
    else:
        bertSimilarity.append( bertSentenceCosineSim(t,t,model_sBERT,mode=0,aggregation="mean_pairs"))
        fasttextSimilarity.append(bertSentenceCosineSim(t,t,model_w2v,mode=1,aggregation="mean_pairs"))
        meanSimSubject.append(np.mean([wmd_gensim(storyData['Lemmas'].iloc[i],storyData['Lemmas'].iloc[j],kv) for j in range(len(storyData['Lemmas'])) if j != i 
                                                                                                and storyData['Subject'].iloc[j] == storyData['Subject'].iloc[i]])) #Subject-wise Word Mover Distance
        meanSimDataset.append(np.mean([wmd_gensim(storyData['Lemmas'].iloc[i],storyData['Lemmas'].iloc[j],kv) for j in range(len(storyData['Lemmas'])) if j != i])) #All Lemmas Word Mover Distance
dfAnalysis['LemEmbBert'] = bertSimilarity
dfAnalysis['LemEmbFtxt'] = fasttextSimilarity


### (4) Semantic Decoupling Extraction
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
import spacy
from storyTools import loadStory,  dictStemmer,lems,preNorm

NLP = spacy.load("pl_core_news_lg")

[storyFrags,partNames] = loadStory(os.path.join(os.getcwd(),"resources","Stories","story3.txt"))
origStory = pd.DataFrame({'fragment': partNames,'StoryFrag': storyFrags})
origStory['StoryLemmas'] = [dictStemmer(origStory['StoryFrag'].iloc[i],morf,norms=preNorm,lems=lems) for i in range(len(origStory))]
origStory['StorySentences'] = [[sent.text for sent in NLP(origStory['StoryFrag'].iloc[i]).sents] for i in range(len(origStory))]

storyData = pd.merge( storyData, origStory,  left_on='fragment', right_on='fragment',  how='left')


semDecoupling_BERT = []; semDecoupling_W2V = []; semDecoupling_WMD = []
for i,(t,u) in enumerate(tqdm(zip(storyData['StoryFrag'].values, storyData['Utterance'].values), total=len(storyData), desc="(4) current fragment")):
     if type(u) is not str:
        semDecoupling_BERT.append(np.nan);semDecoupling_W2V.append(np.nan);semDecoupling_WMD.append(np.nan)
     else:
        semDecoupling_BERT.append(bertSentenceCosineSim(storyData['StorySentences'].iloc[i],storyData['Sentences'].iloc[i],model_sBERT,mode=0,aggregation = 'max'))
        semDecoupling_W2V.append(bertSentenceCosineSim(storyData['StoryLemmas'].iloc[i],storyData['Lemmas'].iloc[i],model_w2v,mode=1,aggregation = 'max'))
        semDecoupling_WMD.append(wmd_gensim(storyData['StoryLemmas'].iloc[i],storyData['Lemmas'].iloc[i],kv))

dfAnalysis['semDecoupling_BERT'] = semDecoupling_BERT   
dfAnalysis['semDecoupling_W2V'] = semDecoupling_W2V  
dfAnalysis['semDecoupling_WMD'] = semDecoupling_WMD    


### (5) Semantic Decoupling between the current thought and NEIGHBOURING fragments (lags -4..+4):
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
def hasText(u):
    return isinstance(u,str) and u.strip() != ""

# Position of a fragment in the story (fixed order for all participants, = trialNum):
# KAROLINA_1 -> 0, JANEK_1 -> 1, KAROLINA_2 -> 2, JANEK_2 -> 3, ...
def storyPos(frag):
    char, num = frag.strip().split('_')
    return 2*(int(num)-1) + int(char == 'JANEK')

origStory['StoryPos'] = origStory['fragment'].apply(storyPos)
storyData['StoryPos'] = storyData['fragment'].apply(storyPos)
assert origStory['StoryPos'].is_unique and len(origStory) == 40, "Unexpected story3.txt structure"

# Story fragment lookup by position - built from the story itself, independent of missing recalls
storyLookup = origStory.set_index('StoryPos')[['StorySentences','StoryLemmas']]
shifts = {'p4': -4, 'p3': -3, 'p2': -2, 'p1': -1, 'n1': +1, 'n2': +2, 'n3': +3, 'n4': +4}
semDecShift = {f"semDecoupling_{m}_{k}": [] for m in ['BERT','W2V','WMD'] for k in shifts}

for i,u in enumerate(tqdm(storyData['Utterance'].values, desc="(5) lags -4..+4")):
    for k,s in shifts.items():
        key = storyData['StoryPos'].iloc[i] + s
        if not hasText(u) or key not in storyLookup.index:   # No utterance or no previous/next fragment -> NaN
            semDecShift[f"semDecoupling_BERT_{k}"].append(np.nan); semDecShift[f"semDecoupling_W2V_{k}"].append(np.nan); semDecShift[f"semDecoupling_WMD_{k}"].append(np.nan)
        else:
            # Neighbouring STORY fragment vs CURRENT utterance
            semDecShift[f"semDecoupling_BERT_{k}"].append(bertSentenceCosineSim(storyLookup.loc[key,'StorySentences'],storyData['Sentences'].iloc[i],model_sBERT,mode=0,aggregation = 'max'))
            semDecShift[f"semDecoupling_W2V_{k}"].append(bertSentenceCosineSim(storyLookup.loc[key,'StoryLemmas'],storyData['Lemmas'].iloc[i],model_w2v,mode=1,aggregation = 'max'))
            semDecShift[f"semDecoupling_WMD_{k}"].append(wmd_gensim(storyLookup.loc[key,'StoryLemmas'],storyData['Lemmas'].iloc[i],kv))

for col,vals in semDecShift.items():
    storyData[col] = vals


### (6) Semantic Decoupling between the current thought and the WHOLE STORY (centroid, without the current fragment):
print(f"[{time.strftime('%H:%M:%S')}] START (5) lags", flush=True)
excl   = 0                                  # 0 = drop only the current fragment; 2 = also drop +-1, +-2
allPos = sorted(storyLookup.index)

semDecWhole = {f"semDecoupling_{m}_whole": [] for m in ['BERT','W2V','WMD']}
for i,u in enumerate(tqdm(storyData['Utterance'].values, desc="(6) whole story")):
    if not hasText(u):
        for col in semDecWhole: semDecWhole[col].append(np.nan)
        continue
    pos  = storyData['StoryPos'].iloc[i]
    keep = [p for p in allPos if abs(p - pos) > excl]
    wholeSentences = [s for p in keep for s in storyLookup.loc[p,'StorySentences']]
    wholeLemmas    = [l for p in keep for l in storyLookup.loc[p,'StoryLemmas']]

    semDecWhole['semDecoupling_BERT_whole'].append(bertSentenceCosineSim(wholeSentences,storyData['Sentences'].iloc[i],model_sBERT,mode=0,aggregation = 'centroid'))
    semDecWhole['semDecoupling_W2V_whole'].append(bertSentenceCosineSim(wholeLemmas,storyData['Lemmas'].iloc[i],model_w2v,mode=1,aggregation = 'centroid'))
    semDecWhole['semDecoupling_WMD_whole'].append(wmd_gensim(wholeLemmas,storyData['Lemmas'].iloc[i],kv))

for col,vals in semDecWhole.items():
    storyData[col] = vals


### Attach the new columns to dfAnalysis BY KEY (Subject + fragment), not by row order
# Sanity check: all earlier columns (LexDiv ... semDecoupling_*) were assigned to dfAnalysis by ROW ORDER,
# so dfAnalysis and storyData must contain the same trials in the same order:
assert len(dfAnalysis) == len(storyData), f"Row count mismatch: dfAnalysis {len(dfAnalysis)} vs storyData {len(storyData)}"
assert (dfAnalysis[['Subject','fragment']].values == storyData[['Subject','fragment']].values).all(), \
       "dfAnalysis and storyData are not in the same row order - positionally assigned columns would be misaligned"

newCols = ['StoryPos'] + list(semDecShift) + list(semDecWhole)
dfAnalysis = dfAnalysis.merge(storyData[['Subject','fragment'] + newCols], on=['Subject','fragment'], how='left', validate='one_to_one')
assert len(dfAnalysis) == len(storyData), "Rows changed during the merge of new columns"



### Save Data:

### Save Data:
# The Attention column was added to storyAnalysis.xlsx by hand - copy it over from the existing file (by key)
oldPath = os.path.join(dataFolder,"storyAnalysis.xlsx")
if 'Attention' not in dfAnalysis.columns and os.path.isfile(oldPath):
    oldAtt = pd.read_excel(oldPath)[['Subject','fragment','Attention']]
    dfAnalysis = dfAnalysis.merge(oldAtt, on=['Subject','fragment'], how='left', validate='one_to_one')
    print(f"Attention copied from {oldPath}: {dfAnalysis['Attention'].notna().sum()} / {len(dfAnalysis)} rows")

# Save as a NEW file - the original storyAnalysis.xlsx stays untouched
outPath = os.path.join(dataFolder,"storyAnalysis_lags.xlsx")
dfAnalysis.to_excel(outPath,index=False)
print(f"Saved: {outPath}  ({len(dfAnalysis)} rows, {dfAnalysis.shape[1]} columns)")



# ### Save Data:

# if os.path.isfile(os.path.join(os.getcwd(),"Results",analysisName,"storyAnalysis.xlsx")):
#     print("Analysis File already! \n")
#     newFileName = input("Input new file name or press Enter to overwrite the data in this file, or Ctrl+C to cancel the analysis")
#     os.makedirs(os.path.join(os.getcwd(),"Results",analysisName), exist_ok=True)
# else:
#     newFileName = ""

# if newFileName != "":
#     dfAnalysis.to_excel(os.path.join(os.getcwd(),"Results",analysisName,newFileName),index=False)
# else:
#     dfAnalysis.to_excel(os.path.join(dataFolder,"storyAnalysis.xlsx"),index=False)