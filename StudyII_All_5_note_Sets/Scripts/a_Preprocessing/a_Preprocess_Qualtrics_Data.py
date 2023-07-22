from Shared_Scripts.Preprocess_Qualtrics_Data import preprocess_qualtrics
from StudyII_All_5_note_Sets.paths import *

# Columns from the qualtrics to be renamed
rename_schema = {
    'sub': 'subject',
    'Q1': 'subject_age',
    'Q2': 'subject_gender',
    'Q3': 'Would you consider yourself a musical person? (0-10)',
    'Q4': 'How many years of formal musical training do you have?',
    'Q5': 'Do you play any instruments (with or without formal training)?',
    'Q6': 'Which instruments do you play (if any)?',
    'Q8_1': 'I understood the instructions of the task',
    'Q8_2': 'I found the task difficult',
    'Q9': 'What general strategy (if any) did you use to judge the similarity across the melodies?',
    'Q10': 'Did this strategy change throughout the task?',
    'Q11': 'How did you listen to the task?',
    'Q13': 'If you have had musical training, how old were you during this period (i.e., age-range)?',
    'Q14': 'What is your first language?',
    'Q15': 'Did you notice any change in the music? Elaborate',
    'Q16': 'How did you arrive at this experiment?',
    'Q18': 'What is your first language (if other)?',
    'Q19': 'Did you notice any particular changes in the music? In other words, did you base your '
           'discriminations on specific features of the melodies?',
    'Q20': 'How did your strategy change throughout the task?',
}

# Columns from the qualtrics to be dropped
drop_schema = [
    'Q3_NPS_GROUP',
    'Q17',
    'DistributionChannel',
    'Duration (in seconds)',
    'EndDate',
    'ExternalReference',
    'Finished',
    'IPAddress',
    'LocationLatitude',
    'LocationLongitude',
    'Progress',
    'RecipientEmail',
    'RecipientFirstName',
    'RecipientLastName',
    'ResponseId',
    'StartDate',
    'Status',
    'UserLanguage',
    "Q9 - Parent Topics",
    "Q9 - Topics",
    "Q16.1",
]

preprocess_qualtrics(qualtrics_dir, SUBJECT_PATTERN, qualtrics_processed_path, rename_schema, drop_schema)
